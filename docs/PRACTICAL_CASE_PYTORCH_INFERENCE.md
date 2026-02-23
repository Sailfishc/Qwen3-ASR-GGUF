# 实战案例：用 Qwen3-ASR-0.6B 转录一段录音

> 本文记录了一次完整的上手过程：从一个 `.webm` 格式的录音文件出发，
> 经过环境排查、依赖安装、模型下载，最终用项目内置的 `qwen_asr` 包
> 完成转录——并借此理解整个项目的设计意图。

---

## 背景

**目标文件**：`/Users/zhangcheng/Downloads/recording-2026-02-23T14-11-34-581Z-d46e76c4.webm`
（约 16 秒的语音，WebM/Opus 格式，251KB）

**目标**：用项目中的 Qwen3-ASR-0.6B 模型进行转录，同时理解项目运作原理。

---

## 第一步：看清项目有什么

首先浏览项目根目录，找到两套并行存在的代码：

```
Qwen3-ASR-GGUF/
├── qwen_asr/              ← 官方 PyTorch 推理包（transformers 后端）
│   └── inference/
│       └── qwen3_asr.py   → Qwen3ASRModel 类
│
├── qwen_asr_gguf/         ← 本项目核心：GGUF 高效推理包
│   └── inference/
│       ├── asr.py         → QwenASREngine（多进程引擎）
│       ├── encoder.py     → ONNX 编码器
│       ├── llama.py       → llama.cpp Python 绑定（ctypes）
│       └── bin/           → (需手动放入) libllama.dylib 等预编译库
│
├── 00~07-*.py             ← 模型转换流水线脚本
├── 11~17-*.py             ← Aligner 转换脚本
├── transcribe.py          ← GGUF 路径命令行工具（主推）
└── transcribe_official.py ← PyTorch 路径演示脚本
```

**关键发现**：这个项目有两条推理路径，本质上是同一个模型的两种运行方式：

| 路径 | 文件 | 依赖 | 模型大小 |
|------|------|------|----------|
| PyTorch（官方） | `qwen_asr/` | torch + transformers | 1.8 GB safetensors |
| GGUF（本项目核心） | `qwen_asr_gguf/` | onnxruntime + libllama | ~460 MB |

---

## 第二步：确认音频文件状态

```bash
# 查询 webm 文件信息
ffprobe -v quiet -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    ~/Downloads/recording-...d46e76c4.webm
# 输出：15.899969  →  约 16 秒
```

项目根目录已经存在一个 `test_audio.wav`（499KB，约 16 秒），
说明之前已用 ffmpeg 将 webm 转换为 WAV：

```bash
ffmpeg -i ~/Downloads/recording-...d46e76c4.webm \
       -ar 16000 -ac 1 \
       ./test_audio.wav
```

> **为什么要转成 WAV？**
> 模型要求输入为 16kHz 单声道 PCM 格式。WebM/Opus 是浏览器录音常见格式，
> librosa/soundfile 对它的支持不稳定，提前用 ffmpeg 转换最可靠。

---

## 第三步：排查两条推理路径的可行性

### GGUF 路径（`transcribe.py`）

检查发现 GGUF 路径需要以下条件，但**全部缺失**：

```bash
ls qwen_asr_gguf/inference/bin/
# 不存在 → 没有预编译的 libllama.dylib / libggml.dylib
# （这些二进制库通常打包在 GitHub Release 的发行包里）

ls model/
# 不存在 → 没有以下模型文件：
#   qwen3_asr_llm.q4_k.gguf           (GGUF Decoder, ~400MB)
#   qwen3_asr_encoder_frontend.int4.onnx
#   qwen3_asr_encoder_backend.int4.onnx
```

**结论**：GGUF 路径需要先下载预编译包和转换好的模型，本次跳过。

### PyTorch 路径（`qwen_asr` 包）

```bash
python3 -c "import torch"
# ModuleNotFoundError: No module named 'torch'
# → 需要安装依赖
```

**结论**：PyTorch 路径缺依赖，但可以安装后使用。

---

## 第四步：安装 Python 依赖

```bash
# 安装 PyTorch（macOS 版本，CPU推理）
pip3 install torch torchaudio --break-system-packages

# 安装项目其他依赖（用 CPU 版 onnxruntime，macOS 不支持 directml）
pip3 install onnxruntime librosa pydub srt typer rich \
             nagisa sentencepiece accelerate \
             --break-system-packages
```

> **注意**：`requirements.txt` 里写的是 `onnxruntime-directml`，
> 这是 Windows 专用（DirectML 是微软的 GPU 加速 API）。
> macOS 需改用 `onnxruntime` 或 `onnxruntime-silicon`（Apple Silicon 优化版）。

---

## 第五步：解决 transformers 版本冲突

安装依赖后尝试导入，报错：

```
ImportError: cannot import name 'check_model_inputs'
from 'transformers.utils.generic'
```

**原因**：pip 自动安装了最新的 `transformers==5.2.0`，
但项目的 `qwen_asr` 包使用了 `transformers==4.57.6` 的内部 API，
两个版本之间接口已发生变化。

**解决**：

```bash
pip3 install "transformers==4.57.6" --break-system-packages
```

> **教训**：看到 `requirements.txt` 有版本锁定（`transformers==4.57.6`），
> 要严格遵守，不能让 pip 自动升级到最新版。

---

## 第六步：尝试复用已缓存的 MLX 模型（失败）

检查 HuggingFace 缓存目录，发现已有一个相关模型：

```bash
ls ~/.cache/huggingface/hub/
# → models--mlx-community--Qwen3-ASR-0.6B-8bit
```

这是 Apple Silicon 的 MLX 格式版本，架构完全相同（`Qwen3ASRForConditionalGeneration`）。
尝试直接用 PyTorch 加载：

```python
Qwen3ASRModel.from_pretrained(mlx_model_path, device_map='cpu', dtype=torch.float32)
# ValueError: The model's quantization config from the arguments has no `quant_method` attribute.
```

**为什么失败？**
MLX 使用自研的量化格式（`"mode": "affine"`），权重存储方式与 PyTorch 的量化方案不兼容，
`transformers` 无法识别。

**结论**：MLX 模型只能用 `mlx` 框架加载，不能直接给 PyTorch 用。

---

## 第七步：下载官方 HuggingFace 模型

需要下载原始的浮点精度模型：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    'Qwen/Qwen3-ASR-0.6B',
    local_dir='./hf_model',
    ignore_patterns=['*.gitattributes', '*.md'],
)
```

**下载后的文件结构**：

```
hf_model/
├── config.json             (模型结构定义，含 model_type: "qwen3_asr")
├── preprocessor_config.json (音频预处理参数：采样率、hop_length 等)
├── tokenizer_config.json   (分词器配置)
├── vocab.json              (词表，2.6MB)
├── merges.txt              (BPE 合并规则，1.6MB)
├── chat_template.json      (对话模板：system/user/assistant 格式)
├── generation_config.json  (生成参数默认值)
└── model.safetensors       (模型权重，1.8GB，float32)
```

> **为什么是 1.8GB？**
> 原始模型是 float32 精度，0.6B 参数 × 4字节 ≈ 2.4GB，
> 但音频编码器和语言解码器共享一些参数，实际 1.8GB。
> 这也是为什么 GGUF 方案把它压缩到 ~460MB 意义重大。

---

## 第八步：执行转录

```python
import sys, time, torch
from pathlib import Path

sys.path.insert(0, str(Path('.').absolute()))   # 让 Python 找到 qwen_asr 包

from qwen_asr import Qwen3ASRModel

# 1. 加载模型（约 3.8 秒）
asr = Qwen3ASRModel.from_pretrained(
    './hf_model',
    device_map='cpu',       # macOS CPU 推理
    dtype=torch.float32,
    low_cpu_mem_usage=True  # 逐层加载，避免内存峰值翻倍
)

# 2. 执行转录（约 6.8 秒）
results = asr.transcribe(
    audio='./test_audio.wav',
    language=None,              # None = 自动识别语言
    return_time_stamps=False    # 不需要字符级时间戳
)

# 3. 打印结果
for r in results:
    print(f'语言: {r.language}')
    print(f'文本: {r.text}')
```

**输出结果**：

```
语言: Chinese
文本: OK，那应该是有已经有了足够的信息，呃，为搜索端写一份架构文档，让我们快速地理解这个项目。
```

**性能数据**：

| 指标 | 数值 |
|------|------|
| 音频时长 | 16 秒 |
| 模型加载 | 3.8 秒 |
| 转录耗时 | 6.8 秒 |
| RTF（实时率） | 0.43（即处理速度约为实时的 2.3 倍） |
| 设备 | macOS CPU（Apple M 系列） |

> **RTF 是什么？**
> RTF (Real-Time Factor) = 处理耗时 / 音频时长。
> RTF = 0.43 表示每秒音频需要 0.43 秒来处理，小于 1 说明比实时快。
> GGUF 方案的目标 RTF ≈ 0.15，比 PyTorch 路径快约 3 倍。

---

## 第九步：理解整个数据流

下图展示了这次转录的数据流，以及 PyTorch 路径和 GGUF 路径的区别：

```
test_audio.wav (16kHz, mono PCM)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  Qwen3ASRProcessor                       │
│  1. librosa 读取音频                                      │
│  2. 提取 80 维 Mel 频谱 (hop=160, win=400)               │
│  3. 拼接 chat_template 构建 prompt                        │
└────────────────────┬────────────────────────────────────┘
                     │
           ┌─────────┴──────────┐
           │ PyTorch 路径        │   GGUF 路径（本项目核心）
           │                    │
           ▼                    ▼
   ┌──────────────┐    ┌─────────────────────┐
   │ Transformer  │    │ ONNX Encoder        │
   │ Encoder 层   │    │ (frontend + backend) │
   │ (在大模型内) │    │ DirectML/CPU 加速    │
   └──────┬───────┘    └──────────┬──────────┘
          │                       │ audio_embedding (float32 向量)
          │                       ▼
          │            ┌──────────────────────┐
          │            │  llama.cpp Decoder   │
          │            │  (GGUF q4_k, ~400MB) │
          │            │  通过 ctypes 调用     │
          │            └──────────┬───────────┘
          │                       │
          └──────────┬────────────┘
                     ▼
            自回归 Token 生成
            （贪心 / 采样）
                     │
                     ▼
            "language Chinese<asr_text>
             OK，那应该是……"
                     │
                     ▼
            parse_asr_output() 解析
            → language="Chinese"
            → text="OK，那应该是……"
```

---

## 总结：这个项目解决了什么问题

| 维度 | 官方 PyTorch 方案 | 本项目 GGUF 方案 |
|------|-------------------|-----------------|
| 模型大小 | 1.8 GB (fp32) | ~460 MB (int4) |
| 依赖 | torch + transformers | onnxruntime + libllama |
| 安装难度 | 中（torch 较大） | 低（可打包发布） |
| 推理速度 (CPU) | RTF ≈ 0.43 | RTF ≈ 0.15 |
| 时间戳支持 | 需要 ForcedAligner | 内置 Aligner 引擎 |
| 跨平台加速 | CUDA/MPS | DirectML/Vulkan/CPU |
| 目标场景 | 研究/服务器 | 桌面端/边缘部署 |

**本次走通的是 PyTorch 路径**，适合快速验证模型效果。
要体验 GGUF 路径的完整性能，需要从 GitHub Releases 下载预编译包（含 `libllama.dylib` 和量化好的 `.gguf`/`.onnx` 模型文件）。

---

## 附：完整复现命令

```bash
# 环境准备
cd /Users/zhangcheng/CodeProjects/Qwen3-ASR-GGUF
pip3 install torch torchaudio --break-system-packages
pip3 install onnxruntime librosa pydub srt typer rich \
             nagisa sentencepiece accelerate --break-system-packages
pip3 install "transformers==4.57.6" --break-system-packages  # 注意版本！

# 音频转换（如果还没有 test_audio.wav）
ffmpeg -i ~/Downloads/recording-*.webm -ar 16000 -ac 1 ./test_audio.wav

# 下载模型
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-ASR-0.6B', local_dir='./hf_model',
                  ignore_patterns=['*.gitattributes', '*.md'])
"

# 执行转录
python3 - <<'EOF'
import sys, torch
sys.path.insert(0, '.')
from qwen_asr import Qwen3ASRModel

asr = Qwen3ASRModel.from_pretrained('./hf_model', device_map='cpu',
                                    dtype=torch.float32, low_cpu_mem_usage=True)
results = asr.transcribe(audio='./test_audio.wav', language=None)
for r in results:
    print(f'[{r.language}] {r.text}')
EOF
```
