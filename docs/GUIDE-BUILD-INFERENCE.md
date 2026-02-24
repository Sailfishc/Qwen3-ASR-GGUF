# 从零构建 ASR 推理脚本：新手导师指南

> **本文定位**：假设你是一个会写 Python、懂基础机器学习概念，但从未接触过 ASR 工程的新手。本文以"导师带徒弟"的方式，一步步带你理解并复刻这个项目的推理脚本。
>
> **最终目标**：读完本文后，你能清楚地回答——"这个项目的每一行代码，为什么要这样写？"
>
> 最后更新：2026-02-24

---

## 目录

1. [本文的学习方式](#1-本文的学习方式)
2. [前置知识体系](#2-前置知识体系)
3. [全局架构：先把地图看清楚](#3-全局架构先把地图看清楚)
4. [第一步：加载音频](#4-第一步加载音频)
5. [第二步：提取 Mel 特征](#5-第二步提取-mel-特征)
6. [第三步：运行 ONNX 编码器](#6-第三步运行-onnx-编码器)
7. [第四步：调用 llama.cpp（ctypes 绑定）](#7-第四步调用-llamacppctypes-绑定)
8. [第五步：构建 Prompt Embedding（最关键）](#8-第五步构建-prompt-embedding最关键)
9. [第六步：LLM 解码循环](#9-第六步llm-解码循环)
10. [第七步：长音频的流式分块](#10-第七步长音频的流式分块)
11. [第八步：多进程架构](#11-第八步多进程架构)
12. [关键难点汇总](#12-关键难点汇总)
13. [调试方法与实践建议](#13-调试方法与实践建议)

---

## 1. 本文的学习方式

### 1.1 "为什么"优先于"是什么"

很多教程告诉你"怎么做"，却不告诉你"为什么这样做"。本文的每个步骤都会先回答：

> **如果你不这样做，会发生什么？**

这是真正理解代码的钥匙。

### 1.2 对照项目文件阅读

本文每个章节都会标注对应的项目文件和行号，建议你一边读本文，一边打开对应代码。

```
本文章节       → 对应项目文件
第4步（音频）  → qwen_asr_gguf/inference/utils.py
第5步（Mel）   → qwen_asr_gguf/inference/encoder.py:8-107
第6步（ONNX）  → qwen_asr_gguf/inference/encoder.py:119-227
第7步（ctypes）→ qwen_asr_gguf/inference/llama.py
第8步（Prompt）→ qwen_asr_gguf/inference/asr.py:80-104
第9步（解码）  → qwen_asr_gguf/inference/asr.py:106-191
第10步（分块） → qwen_asr_gguf/inference/asr.py:269-357
第11步（多进程）→ qwen_asr_gguf/inference/asr_worker.py
```

### 1.3 建议的学习节奏

- **第 2 章**（前置知识）：每个小节读完后，打开对应文件验证你的理解
- **第 3 章**（架构）：重点，至少看 3 遍数据流图
- **第 4-9 章**（核心构建）：逐步阅读，每步都运行一次代码验证
- **第 10-11 章**（流式分块、多进程）：属于进阶内容，理解原理即可

---

## 2. 前置知识体系

在动手写代码之前，你需要先建立 7 个知识模块。每个模块后面都有**"够用就行"的简化版理解**。

---

### 2.1 数字音频基础

**真实的声音**是空气压强随时间变化的连续波。麦克风把它变成电信号，声卡每秒采样 N 次，把连续信号变成离散数字序列——这就是 PCM（脉冲编码调制）。

**关键参数：**

| 参数 | 本项目的值 | 含义 |
|------|-----------|------|
| 采样率 (sample rate) | 16000 Hz | 每秒采 16000 个点 |
| 声道数 (channels) | 1（单声道） | 只保留一路信号 |
| 数据类型 | float32 | 振幅归一化到 [-1.0, 1.0] |

**够用就行的理解：**

> 音频文件 = 一个长度为 `(秒数 × 16000)` 的 float32 数组。16 秒音频 = 256000 个数字。

**❓ 为什么是 16000 Hz，不是 44100 Hz（CD 音质）？**

人声的有效频率范围是 80Hz ~ 8000Hz。奈奎斯特定理告诉我们，采样率 = 2 × 最高频率就够了。16000 Hz 正好覆盖语音，同时数据量比 44100 Hz 少了 63%。Whisper、Qwen-ASR 等模型都用这个采样率。

---

### 2.2 为什么不直接用波形做推理？

你可能会问：音频已经是数字了，为什么不直接把这个数字序列喂给神经网络？

**问题在于：两段相同内容的语音，波形可能完全不同。**

- 说话快慢不同 → 波形时长不同
- 音量大小不同 → 波形幅度不同
- 背景噪声不同 → 波形形状不同

神经网络很难从原始波形中学到稳定的语音特征。

**解决方案：把波形转换为"特征图"（Mel 频谱）。**

这就引出了下一节。

---

### 2.3 Mel 频谱：把声音变成"图片"

**直觉理解：**

把音频想象成一段音乐。你的耳朵会同时感知低音、中音、高音——不同频率的成分。Mel 频谱就是把这种"频率随时间变化"的信息可视化成一张 2D 图：

```
纵轴（128 个 Mel 频率通道）
^
|  ████░░░░░░  ← 低频（bass）
|  ░████░░░░░  ← 中频（speech）
|  ░░░░░████░  ← 高频
+──────────────────────> 横轴（时间帧）
```

每一列（时间帧）= 一个 10ms 的音频片段的频率快照。
每一行（Mel 通道）= 人耳感知的一个频段。

**计算步骤（理解即可，不用记公式）：**

1. **加窗**：取一小段（25ms 的 Hann 窗）音频
2. **FFT**：把时域波形变成频域（各频率的强度）
3. **Mel 映射**：把线性频率映射到 Mel 刻度（模拟人耳对频率的非线性感知）
4. **取对数**：人耳对音量的感知是对数的
5. **归一化**：缩放到合理范围

**本项目的具体参数：**

```
n_mels = 128          # 128 个 Mel 频率通道
hop_length = 160      # 每 160 个采样点（10ms）取一帧
n_fft = 400           # 每帧的 FFT 窗口大小（25ms）
```

**够用就行的理解：**

> Mel 频谱把 `(N 个采样点,)` 的音频变成 `(128, T)` 的 2D 矩阵，其中 T = 音频时长（秒）× 100（每秒 100 帧）。16 秒音频 → `(128, 1600)` 的矩阵。

**对应代码：** `encoder.py:8-107`（`FastWhisperMel` 类）

---

### 2.4 Encoder-Decoder ASR 架构

现代 ASR 系统（Whisper、Qwen-ASR 等）都是 Encoder-Decoder 结构：

```
音频 Mel 特征  →  [Encoder]  →  音频语义向量  →  [Decoder LLM]  →  文字
  (128, T)         编码器        (T', 1536)         解码器
```

**Encoder**（编码器）：把 Mel 频谱（声学特征）压缩成紧凑的语义向量。它"理解"声音中包含的信息。

**Decoder**（解码器）：是一个自回归语言模型，接收音频语义向量作为额外输入，逐词（token）生成文字。

**够用就行的理解：**

> Encoder = 耳朵（把声音变成"意思"）
> Decoder = 嘴巴（把"意思"说成文字）

**本项目的特殊设计：**

- Encoder 用 **ONNX 格式**运行（在 CPU 或 DirectML 上）
- Decoder 是 **GGUF 格式的 LLM**，用 llama.cpp 运行（在 Metal GPU 上）

---

### 2.5 ONNX：模型的"通用接口"

PyTorch 模型像是某个品牌的发动机，只能装在对应品牌的车上。ONNX（Open Neural Network Exchange）是一个标准格式，让模型可以在不同框架上运行。

```
PyTorch 模型  →  [导出]  →  .onnx 文件  →  [onnxruntime]  →  推理
```

**为什么用 ONNX 做 Encoder？**

1. 不需要安装 PyTorch（省掉 1.8GB 依赖）
2. 可以在 CPU 上高效运行（有 AVX 优化）
3. Windows 上可以用 DirectML，利用任意 GPU（不限 NVIDIA）

**够用就行的理解：**

> ONNX 文件 = 模型的"可执行文件"，跨平台，比 PyTorch 轻很多。
> `onnxruntime` = 运行 ONNX 文件的引擎（类比 JVM 运行 .jar 文件）。

---

### 2.6 GGUF 和 llama.cpp：量化的 LLM 推理

LLM（大语言模型）参数通常是 float32（每个参数 4 字节）。Qwen3-ASR-0.6B 有 6 亿参数，这意味着：

- 原始大小：6亿 × 4 字节 = **2.4 GB**
- 量化到 INT4 后：6亿 × 0.5 字节 = **~300 MB**

**量化（Quantization）**：把 float32 权重压缩到更小的位宽（INT8、INT4等），以较小的精度损失换取极大的内存减少和速度提升。

**GGUF** 是 llama.cpp 项目定义的量化模型格式（`.gguf` 文件）。

**llama.cpp** 是用 C++ 写的高效 LLM 推理引擎，支持：
- Metal（Apple GPU）加速
- 极低的内存占用
- 不依赖 CUDA

**够用就行的理解：**

> GGUF = 量化压缩后的 LLM 文件格式
> llama.cpp = 运行 GGUF 文件的引擎，本项目用它做 Decoder

---

### 2.7 Python ctypes：Python 调用 C 库

llama.cpp 是 C++ 写的，编译后是一个 `.dylib`（macOS）/ `.dll`（Windows）/ `.so`（Linux）动态库。

Python 没法直接用 C++ 代码，但可以用 `ctypes` 模块调用编译好的 C 接口。

**基本用法（简化版）：**

```python
import ctypes

# 1. 加载动态库
lib = ctypes.CDLL("libllama.dylib")

# 2. 声明函数签名（输入/输出类型）
lib.llama_backend_init.argtypes = []
lib.llama_backend_init.restype = None

# 3. 调用函数
lib.llama_backend_init()
```

**为什么要声明类型？**

C 函数没有运行时类型信息，Python 必须提前告诉 ctypes "这个函数接受什么参数、返回什么类型"，否则内存解释会出错，直接导致程序崩溃（segfault）。

**够用就行的理解：**

> ctypes = Python 和 C 库之间的"翻译官"，必须精确描述函数签名，否则会崩溃。

**对应代码：** `llama.py:156-350`（`init_llama_lib` 函数）

---

## 3. 全局架构：先把地图看清楚

在写任何代码之前，必须搞清楚数据是怎么流动的。这是整个项目的核心数据流：

```
音频文件 (m4a/mp3/wav)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤 1: 加载音频                                            │
│  pydub → numpy array                                        │
│  输出：float32 数组，形状 (N,)，N = 秒数 × 16000            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤 2: 提取 Mel 频谱                                       │
│  FastWhisperMel                                             │
│  输出：float16/32 矩阵，形状 (128, T)，T ≈ 秒数 × 100      │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴─────────────┐
         ▼  (在子进程中运行)         │
┌────────────────────┐              │
│  步骤 3a: ONNX     │              │
│  Frontend          │              │
│  (100帧一块循环)    │              │
│  输出：(T', D_fe)  │              │
└────────┬───────────┘              │
         ▼                          │
┌────────────────────┐              │
│  步骤 3b: ONNX     │              │
│  Backend           │              │
│  (Transformer)     │              │
│  输出：(T', 1536)  │              │
└────────┬───────────┘              │
         │  通过 multiprocessing.Queue 传回主进程
         ▼                          │
┌─────────────────────────────────────────────────────────────┐
│  步骤 4: 构建 Prompt Embedding                               │
│                                                             │
│  [system token embd]                                        │
│  + [user token embd]                                        │
│  + [audio_start token embd]                                 │
│  + [audio embedding (T', 1536)] ← ONNX 输出直接插入这里     │
│  + [audio_end token embd]                                   │
│  + [assistant token embd]                                   │
│  + [language token embd]                                    │
│  + [历史文本 token embd]                                    │
│                                                             │
│  输出：float32 矩阵，形状 (total_len, 1536)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  步骤 5: LLM 解码                                            │
│  llama.cpp (GGUF 模型)                                      │
│                                                             │
│  Prefill：把整个 Embedding 序列一次性输入，建立 KV Cache     │
│  Generate：循环采样下一个 token，直到 <|im_end|>             │
│                                                             │
│  输出：token ID 序列                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
               token_to_bytes()
                      │
                      ▼
                  UTF-8 文字
```

**架构最重要的设计决策（要记住）：**

1. **ONNX Encoder 跑在子进程**，llama.cpp Decoder 跑在主进程 → 防止两个大型 C 库的内存冲突
2. **音频 Embedding 直接注入 LLM 的 Embedding 空间** → 不需要跨模态适配器（Qwen3-ASR 天生支持）
3. **分块流式处理** → 长音频不会爆内存

---

## 4. 第一步：加载音频

**目标：** 把各种格式的音频文件变成统一的 float32 numpy 数组。

**为什么用 pydub，而不是直接读文件？**

音频格式五花八门（m4a、mp3、opus、flac、wav...），每种都有不同的压缩算法。pydub 底层调用 ffmpeg，能解码所有格式，我们不需要自己处理每种编解码器。

**核心代码逻辑（`utils.py:57-81`）：**

```python
from pydub import AudioSegment
import numpy as np

def load_audio(audio_path, sample_rate=16000):
    # pydub 加载并自动转换为单声道、16kHz
    audio_segment = AudioSegment.from_file(audio_path,
                                           frame_rate=sample_rate,
                                           channels=1)
    # 从 pydub 对象取出整数采样点，归一化到 [-1.0, 1.0]
    bit_depth = audio_segment.sample_width * 8  # 通常是 16 bit
    max_val = float(1 << (bit_depth - 1))       # 16bit = 32768

    audio = np.array(audio_segment.get_array_of_samples()) / max_val
    return audio.astype(np.float32)
```

**动手验证：**

```python
from qwen_asr_gguf.inference.utils import load_audio
audio = load_audio("test_audio.wav")
print(audio.shape)   # → (256000,) 对于16秒音频
print(audio.dtype)   # → float32
print(audio.min(), audio.max())  # → 大约 [-1.0, 1.0] 之间
```

**⚠️ 常见坑：**
- 忘记安装 ffmpeg → pydub 无法解码 m4a/mp3，只能处理 wav
- 音频采样率不是 16kHz → pydub 会自动重采样，但要确认 `frame_rate=16000` 参数传了

---

## 5. 第二步：提取 Mel 特征

**目标：** 把 `(N,)` 的音频数组变成 `(128, T)` 的 Mel 频谱矩阵。

### 5.1 为什么项目自己实现 Mel，而不用 librosa？

`librosa` 是标准音频处理库，有现成的 `librosa.feature.melspectrogram()`。但它有一个大问题：**启动时用 numba JIT 编译，冷启动需要 5-10 秒**。在频繁调用的推理场景下体验极差。

项目里的 `FastWhisperMel` 完全用 numpy + scipy 实现，冷启动时间接近零。

### 5.2 Mel 提取的代码逻辑（精简版）

对应 `encoder.py:76-107`（`__call__` 方法）：

```python
# 1. 给音频两端补零（保持边缘帧完整性）
y = np.pad(audio, n_fft // 2, mode='reflect')

# 2. 分帧（利用 stride_tricks 零拷贝切片，非常高效）
#    每个帧 = n_fft=400 个采样点
#    帧步长 = hop_length=160 个采样点（10ms）
frames = np.lib.stride_tricks.as_strided(y, ...)

# 3. 加窗（Hann 窗）并做 FFT
stft = np.fft.rfft(frames * hann_window, axis=0)

# 4. 计算能量谱
magnitudes = np.abs(stft) ** 2

# 5. 通过 Mel 滤波器组（128行×201列的矩阵乘法）
mel_spec = np.dot(filters.T, magnitudes)

# 6. 取对数 + 归一化（Whisper 的标准做法）
log_spec = np.log10(np.maximum(mel_spec, 1e-10))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

**关键点：输出精度**

ONNX 模型接受 float16 输入（比 float32 内存少一半）：

```python
# encoder.py:149
fe_input_type = self.sess_fe.get_inputs()[0].type
self.input_dtype = np.float16 if 'float16' in fe_input_type else np.float32
```

Mel 提取时就要生成对应精度的数组，避免后续额外的类型转换。

**动手验证：**

```python
from qwen_asr_gguf.inference.encoder import FastWhisperMel
import numpy as np

mel_extractor = FastWhisperMel()
audio = np.random.randn(16000 * 5).astype(np.float32)  # 模拟5秒音频
mel = mel_extractor(audio)
print(mel.shape)   # → (128, 500)  5秒 × 100帧/秒
```

---

## 6. 第三步：运行 ONNX 编码器

这是整个项目**最有工程创意的部分**，也是理解难度最大的地方。

### 6.1 为什么要把 Encoder 分成 Frontend 和 Backend？

原始 Qwen3-ASR 的 Encoder 是一个整体。对于 40 秒音频，Mel 特征大小是 `(128, 4000)`，一次性过 Encoder 需要大量显存。

作者的解决方案：**把 Encoder 拆成两半**。

```
原始 Encoder
    │
    ├── Frontend（卷积层）：处理单个 100 帧窗口
    │   输入：(1, 128, 100)  → 输出：(1, 13, 896)
    │   特点：无上下文依赖，可以无限分块并行/串行处理
    │
    └── Backend（Transformer层）：处理全序列
        输入：(1, T', 896)   → 输出：(1, T', 1536)
        特点：需要完整序列，一次性处理
```

**关键洞察：Frontend 是无状态的局部操作（卷积），Backend 是有状态的全局操作（Attention）。** 所以可以把 Frontend 分块循环处理，拼接后再一次性过 Backend。

### 6.2 100 帧为什么是"100"？

这是模型设计决定的。Frontend 的卷积层在 100 帧窗口上产生固定输出：

- 输入：(1, 128, 100) → 100 帧 Mel
- 输出：(1, **13**, 896) → 13 帧特征

这个 13 是卷积步长决定的，不能改。所以我们的分块必须以 100 帧为单位。

**`get_feat_extract_output_lengths` 函数（`encoder.py:109-117`）：**

```python
def get_feat_extract_output_lengths(input_lengths):
    """计算输入 T 帧 Mel 对应多少帧 Frontend 输出"""
    input_lengths_leave = input_lengths % 100       # 不满100帧的余数
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return int(output_lengths)
```

**为什么需要这个函数？**

Mel 帧数 T 通常不是 100 的整数倍。我们 Pad 到 100 倍数后，Frontend 会多输出几个帧（来自 padding）。这个函数精确计算有效帧数，让我们可以从结果中截掉无效的 padding 输出。

**如果不截掉会怎样？** LLM 会"看到"一些空白的 padding 特征，可能导致乱码或幻觉。

### 6.3 Backend 的 Attention Mask

```python
# encoder.py:198
mask = np.zeros((batch, 1, seq_len, seq_len), dtype=self.input_dtype)
```

Attention Mask 用来控制哪些位置可以互相"看见"。全零表示所有位置之间没有 mask——即每个帧都能关注所有帧。

**❓ 为什么是全零，而不是全一？**

Transformer 的 Attention 计算中，mask 的语义通常是"加到 logit 上"：
- 0 → 不遮挡（正常 attention）
- -∞（或很大负数）→ 遮挡（忽略这个位置）

全零 mask = 不遮挡任何位置 = 完全双向 attention（每帧看所有帧）。这是 Encoder 的标准做法（与 Decoder 的因果 mask 不同）。

### 6.4 Frontend 循环推理

对应 `encoder.py:160-190`（`_run_frontend` 方法）：

```python
# 1. Pad 到 100 的倍数
pad_len = (100 - (T % 100)) % 100
if pad_len > 0:
    mel = np.pad(mel, ((0,0), (0, pad_len)), mode='constant')

# 2. 按 100 帧分块，循环推理
num_chunks = mel_input.shape[2] // 100
fe_outputs = []
for i in range(num_chunks):
    chunk = mel_input[:, :, i*100 : (i+1)*100]
    out = sess_fe.run(None, {"chunk_mel": chunk})[0]  # (1, 13, 896)
    fe_outputs.append(out)

# 3. 拼接所有块的输出
hidden_states = np.concatenate(fe_outputs, axis=1)  # (1, N*13, 896)

# 4. 截取有效长度（去掉 padding 带来的多余帧）
t_out = get_feat_extract_output_lengths(T)
hidden_states = hidden_states[:, :t_out, :]
```

**动手验证：**

```python
import onnxruntime as ort
import numpy as np

sess_fe = ort.InferenceSession("model/qwen3_asr_encoder_frontend.int4.onnx")
# 随机一块 100 帧 Mel
chunk = np.random.randn(1, 128, 100).astype(np.float16)
out = sess_fe.run(None, {"chunk_mel": chunk})[0]
print(out.shape)  # → (1, 13, 896)
```

---

## 7. 第四步：调用 llama.cpp（ctypes 绑定）

这一步是整个项目中**技术门槛最高**的部分。

### 7.1 为什么不用 llama-cpp-python 包？

`llama-cpp-python` 是 llama.cpp 的官方 Python 绑定，有 pip 包，应该最方便。

问题在于：**这个项目使用了 llama.cpp 最新 API 中的字段**，而 pip 上的 `llama-cpp-python` 版本落后，struct 布局不匹配。

具体来说，项目的 `llama_context_params` struct 包含：

```python
# llama.py:82-85（最新字段）
("swa_full", ctypes.c_bool),
("kv_unified", ctypes.c_bool),
("samplers", ctypes.POINTER(ctypes.c_void_p)),
("n_samplers", ctypes.c_size_t),
```

`llama-cpp-python 0.3.16` 的对应 struct 没有这些字段。Python 传入的内存布局和 C 期望的布局不一致，直接导致 **segfault（段错误，程序崩溃）**。

**解决方案：自己编译 llama.cpp，并用 ctypes 手写绑定（精确控制 struct 布局）。**

### 7.2 ctypes 绑定的结构

对应 `llama.py:156-350`（`init_llama_lib` 函数）：

```python
def init_llama_lib():
    # 1. 加载动态库
    llama = ctypes.CDLL("qwen_asr_gguf/inference/bin/libllama.dylib")

    # 2. 逐一声明函数签名
    llama_backend_init = llama.llama_backend_init
    llama_backend_init.argtypes = []
    llama_backend_init.restype = None

    llama_model_load_from_file = llama.llama_model_load_from_file
    llama_model_load_from_file.argtypes = [
        ctypes.c_char_p,                     # 文件路径（C字符串）
        llama_model_params                   # 参数结构体
    ]
    llama_model_load_from_file.restype = ctypes.c_void_p  # 返回模型指针

    # ... 类似地声明其他30多个函数
```

**⚠️ 必须精确匹配 C 函数签名，否则：**

| 错误类型 | 原因 |
|---------|------|
| segfault（崩溃） | argtypes/restype 类型错误，内存越界 |
| 返回值乱码 | restype 声明为错误类型 |
| 参数值异常 | struct 字段顺序/大小错误 |

### 7.3 LlamaModel、LlamaContext、LlamaBatch

项目把 ctypes 调用封装成三个 Python 类（`llama.py:350+`）：

```
LlamaModel   ← 加载 .gguf 文件，管理模型权重
     │
LlamaContext ← 创建推理上下文（KV Cache 在这里）
     │
LlamaBatch   ← 一次推理的输入批次（token IDs 或 embeddings）
```

使用方式（简化版）：

```python
# 加载模型
model = LlamaModel("model/qwen3_asr_llm.q4_k.gguf")

# 创建推理上下文（n_ctx=2048 表示最大上下文长度）
ctx = LlamaContext(model, n_ctx=2048, n_batch=4096)

# 构建输入批次（embedding 模式）
batch = LlamaBatch(max_tokens=1024, embd_dim=model.n_embd, n_seq_max=1)
batch.set_embd(audio_prompt_embd)  # 设置 embedding 输入

# 推理
ctx.decode(batch)

# 采样下一个 token
sampler = LlamaSampler(temperature=0.4)
next_token = sampler.sample(ctx.ptr)
```

---

## 8. 第五步：构建 Prompt Embedding（最关键）

这是整个项目的**核心创新点**，也是理解难度最大的概念。

### 8.1 什么是 Token Embedding？

LLM 的输入是 token（文字片段）的 ID 序列，但神经网络处理的是浮点向量。"Embedding Table"（嵌入矩阵）把每个 token ID 映射成一个高维向量。

```
token ID: 1234  →  [0.12, -0.54, 0.89, ...]  ← 长度 = n_embd = 1536
```

Qwen3-ASR-0.6B 的 `n_embd = 1536`（每个 token 对应 1536 维向量）。

### 8.2 音频特征和文字 token 维度一样！

**这是 Qwen3-ASR 最关键的架构设计：**

ONNX Backend 的输出大小是 `(T', 1536)`——刚好和 token embedding 的维度相同！

这意味着我们可以把音频特征和文字 token embedding **直接拼接在同一序列里**，作为 LLM 的输入。

### 8.3 Prompt 的完整结构

对应 `asr.py:80-104`（`_build_prompt_embd` 方法）：

```
完整的 Prompt Embedding（形状：(total_len, 1536)）

┌────────────────────────────────┐
│ <|im_start|> system            │  ← token embedding（查表）
│ You are a helpful assistant.   │
│ <|im_end|>                     │
├────────────────────────────────┤
│ <|im_start|> user              │  ← token embedding
│ <|audio_start|>                │  ← 特殊token，标记音频开始
├────────────────────────────────┤
│ 音频特征向量 (T', 1536)         │  ← ONNX Backend 的直接输出！
│ （T' 帧，每帧 1536 维）         │    不是token，是连续向量
├────────────────────────────────┤
│ <|audio_end|>                  │  ← token embedding
│ <|im_end|>                     │
├────────────────────────────────┤
│ <|im_start|> assistant         │  ← token embedding
│ language Chinese               │  ← 语言提示（可选）
│ <asr_text>                     │  ← 告诉模型"接下来输出转录文字"
│ [历史分段的文字（记忆）]        │  ← 上一段的转录结果（作为上下文）
└────────────────────────────────┘
```

**构建代码（精简版）：**

```python
# 查 embedding table 获取 token 向量
prefix_tokens = [ID_IM_START, *tokenize("system\n..."), ID_IM_END,
                 ID_IM_START, *tokenize("user\n"), ID_AUDIO_START]

suffix_tokens = [ID_AUDIO_END, ID_IM_END,
                 ID_IM_START, *tokenize("assistant\nlanguage Chinese"),
                 ID_ASR_TEXT, *tokenize(previous_text)]

# 全部拼成一个大矩阵
total_len = len(prefix_tokens) + audio_embd.shape[0] + len(suffix_tokens)
total_embd = np.zeros((total_len, n_embd), dtype=np.float32)

total_embd[:len(prefix_tokens)]           = embedding_table[prefix_tokens]
total_embd[len(prefix_tokens):len(prefix_tokens)+audio_len] = audio_embd  # 直接插入
total_embd[len(prefix_tokens)+audio_len:] = embedding_table[suffix_tokens]
```

**⚠️ 关键细节：数据类型转换**

ONNX Backend 输出的是 float16，但 LLM 的 embedding table 是 float32。必须在拼接前统一类型：

```python
audio_embd = audio_embd.astype(np.float32)  # float16 → float32
```

如果忘了这一步，numpy 的广播规则可能导致精度问题或形状不匹配。

### 8.4 为什么这样设计能工作？

**从模型的角度看：** Qwen3-ASR 在预训练时就专门设计了 `<|audio_start|>` ... `<|audio_end|>` 这两个特殊 token，模型知道这对 token 之间的内容是音频特征。

**关键在于：** 音频特征向量和 token embedding 向量处于同一语义空间（都是 1536 维，来自同一个嵌入体系）。模型的 Attention 可以直接把文字和音频特征关联起来。

**如果是普通 LLM（如 GPT-4）：** 音频维度可能不匹配，需要额外的"投影层"适配。Qwen3-ASR 的架构天生支持这个，是它的核心设计亮点。

---

## 9. 第六步：LLM 解码循环

有了 Prompt Embedding 矩阵，现在要让 LLM 把它"读进去"，然后生成文字。

### 9.1 Prefill 阶段

**Prefill（预填充）**：把整个 Prompt Embedding 一次性输入 LLM，建立 KV Cache。

```python
# asr.py:118-127
batch = LlamaBatch(max_tokens=total_len * 4, embd_dim=model.n_embd)
batch.set_embd(full_embd, pos=pos_arr)  # 设置 embedding 输入

ctx.clear_kv_cache()  # 清空上次的 KV Cache
ctx.decode(batch)     # 执行 Prefill
```

**什么是 KV Cache？**

Transformer 在计算 Attention 时，每个位置要看所有历史位置的 Key 和 Value。KV Cache 把已计算的 K、V 缓存起来，避免生成时重复计算。

Prefill 之后，KV Cache 里存了 Prompt 中所有位置的 K、V 信息（包括音频特征的上下文）。

### 9.2 Generation 阶段（自回归采样）

对应 `asr.py:129-178`：

```python
# 初始化采样器
sampler = LlamaSampler(temperature=0.4, seed=random_seed)

# 从 Prefill 的最后位置采样第一个 token
last_token = sampler.sample(ctx.ptr)

for _ in range(512):  # 最多生成 512 个 token
    # 遇到结束标记就停止
    if last_token in [eos_token, ID_IM_END]:
        break

    # 把这个 token 喂回模型（只有一个 token，很快）
    ctx.decode_token(last_token)

    # 采样下一个 token
    last_token = sampler.sample(ctx.ptr)

    # 把 token ID 转换成字节，再解码为 UTF-8 文字
    piece = token_to_bytes(last_token).decode('utf-8', errors='replace')
    print(piece, end='', flush=True)
```

**Temperature 是什么？**

- temperature=0 → 每次都选概率最大的 token（贪心，输出固定）
- temperature=0.4 → 有一点随机性，输出更自然
- temperature=1.0 → 完全按概率分布采样，输出多样但可能离谱

ASR 任务通常用较低的 temperature（0.4）：希望输出准确，稍微有点容错。

### 9.3 熔断检测：防止重复循环

LLM 有时会陷入"重复输出"的死循环，比如一直输出"的的的的的的..."。项目加了一个简单的熔断机制：

```python
# asr.py:157-161
if len(stable_tokens) > 15:
    if len(set(stable_tokens[-15:])) <= 3:  # 最近15个token，只有3种不同
        result.is_aborted = True
        break
```

**并在外层加了重试（`asr.py:202-208`）：**

```python
for i in range(4):  # 最多重试4次
    res = self._decode(...)
    if not res.is_aborted:
        break
    temperature += 0.3  # 每次重试提高 temperature，增加多样性
```

### 9.4 Rollback Buffer（预留区间）

**为什么要有 `rollback_num=5`？**

因为一个汉字可能对应多个 byte，多个 byte 对应多个 token。如果只生成了半个字的 token，`decode('utf-8')` 会报错。

项目用一个 deque 实现"最多预留 5 个 token 不立即输出"，等 5 个 token 攒够了（基本确定不是一个汉字的分割点），再一起解码输出。

---

## 10. 第七步：长音频的流式分块

到目前为止，我们实现了"处理一段音频"的能力。但现实中需要处理几分钟甚至几小时的音频。

### 10.1 为什么不一次性处理整段音频？

1. **内存**：60 秒音频的 Prompt Embedding 大小约 = 60 × 100 × 1536 × 4 bytes = 360MB，太大了
2. **KV Cache**：n_ctx=2048 限制了最大上下文长度，60 秒 ≈ 6000 帧，超出限制

### 10.2 分块策略

项目把长音频切成 40 秒一块（`chunk_size=40`）：

```
音频时间轴：
──────────────────────────────────────────────────
|  chunk 0  |  chunk 1  |  chunk 2  |  chunk 3  |
|   0~40s   |  40~80s   |  80~120s  | 120~160s  |
──────────────────────────────────────────────────
```

**记忆机制（`memory_chunks=1`）：**

每处理一个 chunk，把上一个 chunk 的转录文字带入下一个 chunk 的 Prompt，作为"记忆"：

```
处理 chunk 1 时的 Prompt：
  [system]
  [audio: chunk 1 的音频特征]
  [assistant] language Chinese
  [asr_text] ← chunk 0 的转录结果（记忆）
```

**为什么要带历史？**

1. 断句更准确（上下文连贯）
2. 避免在 chunk 边界出现语义割裂
3. 提高专有名词的一致性

### 10.3 三级流水线

对应 `asr.py:350-357`，项目实现了一个聪明的流水线：

```
时间线：
t=0  ┤
     │  发送 chunk0 的编码任务 →
t=1  ┤
     │  等待 chunk0 编码完成
     │  同时：发送 chunk1 的编码任务 →
t=2  ┤
     │  chunk0 解码（LLM 生成）    │ chunk1 编码中（子进程）
t=3  ┤
     │  chunk0 对齐（子进程异步）  │ chunk1 解码（LLM 生成）
t=4  ┤
```

**意义：编码（ONNX）和解码（LLM）并行运行，大幅减少等待时间。**

---

## 11. 第八步：多进程架构

### 11.1 为什么要用多进程，而不是多线程？

**Python 的 GIL（全局解释器锁）**限制了同一时刻只有一个线程执行 Python 代码。

但更重要的原因是：**两个 C 库（onnxruntime 和 llama.cpp）在同一进程中可能产生内存冲突**（共享内存分配器、线程池等）。用独立进程，各自有独立的内存空间，彻底隔离。

### 11.2 进程间通信

对应 `asr.py:42-51` 和 `asr_worker.py`：

```
主进程 (llama.cpp)
    │
    │  multiprocessing.Queue（发送命令）
    ▼
子进程 (onnxruntime)
    │
    │  multiprocessing.Queue（返回结果）
    ▼
主进程 (接收 embedding，喂给 LLM)
```

**通信协议（`schema.py:7-16`）：**

```python
class MsgType(Enum):
    CMD_ENCODE = auto()  # 主→子：发送音频，请求编码
    CMD_STOP = auto()    # 主→子：停止子进程
    MSG_EMBD = auto()    # 子→主：返回编码后的 embedding
    MSG_READY = auto()   # 子→主：就绪信号（子进程启动完成）
    MSG_ERROR = auto()   # 子→主：错误信号
```

---

## 12. 关键难点汇总

这里列出整个推理流程中**最容易出错**的地方：

### 12.1 数据类型链条

```
音频 float32
    │ FastWhisperMel
    ▼
Mel 特征 float16（或 float32，取决于 ONNX 模型输入类型）
    │ ONNX Frontend
    ▼
Frontend 输出 float16
    │ ONNX Backend
    ▼
Backend 输出 float16
    │ .astype(np.float32)  ← 必须转换！
    ▼
音频 embedding float32
    │ np.concatenate with token embeddings (float32)
    ▼
完整 Prompt Embedding float32
    │ LlamaBatch.set_embd
    ▼
LLM 输入 float32
```

**❌ 常见错误：** 忘记把 float16 音频 embedding 转换为 float32，导致 dtype 不匹配，numpy 抛出错误或默默向上转型（浪费内存）。

### 12.2 Padding 和有效长度切片

```python
# 错误写法（忘记切片）：
hidden_states = np.concatenate(fe_outputs, axis=1)  # 包含 padding 帧！

# 正确写法：
t_out = get_feat_extract_output_lengths(T)  # 精确计算有效帧数
hidden_states = hidden_states[:, :t_out, :]  # 切掉多余的 padding 帧
```

**为什么这会出问题？** padding 帧是 constant(0) 填充，Frontend 看到它们会产生"无意义"的特征向量。LLM 看到这些向量会感到困惑，可能产生多余的空格或标点。

### 12.3 ctypes struct 字段顺序

ctypes struct 的字段必须和 C struct **完全一致**（字段顺序、类型、大小）。

如果你想更新 llama.cpp 版本，必须同时检查 `llama_context_params` struct 是否有变化。一旦有新字段或字段顺序改变，必须同步更新 `llama.py`。

**验证方法：** 查看 llama.cpp 的 `llama.h` 头文件中 `llama_context_params` 的定义，与 `llama.py:52-86` 对比。

### 12.4 KV Cache 的清空时机

```python
# 正确：每次处理新 chunk 之前清空
ctx.clear_kv_cache()
ctx.decode(batch)

# 错误：忘记清空（KV Cache 里还有上一个 chunk 的内容）
ctx.decode(batch)  # ← 错误！KV Cache 污染
```

每个 chunk 的 Prompt 是独立的（它自己携带了历史文字），所以每次必须从干净的 KV Cache 开始。

### 12.5 多进程里不能用 CUDA / MPS

子进程不会继承主进程的 GPU context。如果 onnxruntime 在主进程里初始化了 GPU，子进程里需要重新初始化。这是作者让 Encoder 在子进程中初始化（不是传递已初始化的对象）的原因。

---

## 13. 调试方法与实践建议

### 13.1 最小验证脚本的写法

每完成一步，就写一个最小脚本验证输出形状：

```python
# 验证第5步（Mel提取）
from qwen_asr_gguf.inference.encoder import FastWhisperMel
import numpy as np

mel = FastWhisperMel()
audio = np.random.randn(16000 * 5).astype(np.float32)
out = mel(audio)
print(f"Mel shape: {out.shape}")       # 应该是 (128, 500)
print(f"Mel dtype: {out.dtype}")       # 应该是 float32
print(f"Mel range: [{out.min():.2f}, {out.max():.2f}]")  # 应该在合理范围内
assert out.shape == (128, 500), "形状错了！"
```

**"先验证形状，再验证值"** 是调试深度学习推理最实用的策略。

### 13.2 在 `21-Run-ASR.py` 中下断点

用 VSCode 或 PyCharm 的调试器，在以下位置设置断点，观察张量的变化：

```python
# 断点1：encoder.py:227（encode 方法的 return 语句之前）
# 观察：audio_embd.shape 应该是 (T', 1536)

# 断点2：asr.py:102（构建 Prompt 之后）
# 观察：total_embd.shape 应该是 (total_len, 1536)
# 观察：n_pre, n_aud, n_suf 各自多大

# 断点3：asr.py:141（生成循环内）
# 观察：每次 last_sampled_token 是什么整数
# 观察：token_to_bytes(last_sampled_token) 解码出什么字
```

### 13.3 打印 shape 是你最好的朋友

在调试推理脚本时，在每个关键操作后加一行 `print(f"step X: shape={array.shape}, dtype={array.dtype}")`。等一切正常了再删掉。

### 13.4 如果遇到 segfault

segfault 通常来自 ctypes 类型声明错误。排查步骤：

1. **锁定最后一次成功的 ctypes 调用**（在每个 ctypes 调用前后加 `print`）
2. **对照 `llama.h`** 检查出问题函数的参数类型
3. **检查 struct 字段**：逐字段对比 `llama.py` 和 `llama.h`

### 13.5 建议的复刻顺序

如果你想从零复刻这个项目，建议按此顺序：

```
阶段1：单独验证每个组件（1天）
  ├── 能加载音频并打印波形
  ├── 能提取 Mel 并可视化（plt.imshow）
  ├── 能独立运行 ONNX Frontend/Backend
  └── 能独立运行 GGUF 解码（纯文字输入）

阶段2：打通音频→文字的最短路径（1-2天）
  ├── 硬编码一段短音频（<5秒）
  ├── 手动构建 Prompt Embedding
  └── 验证 LLM 能输出有意义的文字

阶段3：支持任意长度音频（1天）
  ├── 实现分块逻辑
  └── 添加记忆拼接

阶段4：添加多进程（可选）
  └── 把 Encoder 移到子进程
```

---

## 附录：关键数字速查

| 参数 | 值 | 含义 |
|------|-----|------|
| 采样率 | 16000 | 每秒采样点数 |
| Mel 通道 | 128 | 频率轴维度 |
| 帧率 | 100 | 每秒 Mel 帧数 |
| Frontend 块大小 | 100 帧 | 前端分块单位 |
| Frontend 压缩比 | 100→13 | 每100帧输出13帧 |
| Frontend 输出维度 | 896 | 前端隐层维度 |
| Backend 输出维度 | 1536 | 后端/LLM 嵌入维度 |
| n_embd | 1536 | LLM token embedding 维度 |
| n_ctx | 2048 | LLM 上下文窗口大小 |
| chunk_size | 40 秒 | 分块大小 |
| temperature | 0.4 | 采样温度 |
| max_new_tokens | 512 | 每块最多生成 token 数 |

---

**文档结束**

> 如果你在某一步卡住了，建议先回到第 3 章（全局架构）重新对照数据流图，确认你理解了该步骤"输入什么、输出什么"，再继续。
