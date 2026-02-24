# Qwen3-ASR-GGUF 项目学习计划

> 文档版本：1.0  
> 最后更新：2026-02-23  
> **推荐阅读顺序：第 3 顺位**（在完成架构和集成文档阅读后）

---

## 📋 本文档阅读指南

### 在完整项目文档中的阅读顺序

| 顺位 | 文档 | 文件名 | 目标读者 | 预计耗时 |
|:----:|------|--------|----------|----------|
| **①** | [项目架构文档](./02-ARCHITECTURE.md) | `02-ARCHITECTURE.md` | 想了解项目整体设计 | 1-2 小时 |
| **②** | [集成指南](./03-INTEGRATION.md) | `03-INTEGRATION.md` | 想快速使用项目 | 1-2 小时 |
| **③** | **学习计划** | `06-LEARNING_PLAN.md` | 想深入理解原理 | 4-12 周 |
| **④** | [导出流程](./EXPORT_GUIDE.md) | `EXPORT_GUIDE.md` | 想转换自己的模型 | 2-4 小时 |
| **⑤** | [源码解析](./SOURCE_CODE.md) | `SOURCE_CODE.md` | 想修改/扩展功能 | 4-8 周 |

### 本学习计划使用指南

```
建议学习路径：

基础薄弱          基础较好          时间紧张
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐      ┌─────────┐      ┌─────────┐
│ 阶段一   │      │ 快速回顾 │      │ 阶段二   │
│ 2-4 周   │      │ 1 周     │      │ 直接实战 │
│         │      │         │      │         │
│ 阶段二   │      │ 阶段二   │      │ 边做边学 │
│ 4-6 周   │      │ 4-6 周   │      │         │
│         │      │         │      │         │
│ 阶段三   │      │ 阶段三   │      │ 查阅文档 │
│ 4-8 周   │      │ 4-8 周   │      │ 按需学习 │
└─────────┘      └─────────┘      └─────────┘
```

---

## 🎯 学习目标

完成本学习计划后，你将能够：

| 目标层级 | 能力描述 |
|---------|----------|
| **基础** | 独立部署和使用 Qwen3-ASR 进行语音转录 |
| **进阶** | 理解模型导出和量化流程，可转换自己的模型 |
| **高级** | 修改和优化代码，适配新场景或新模型 |
| **专家** | 从头实现类似的语音识别系统 |

---

## 📊 前置知识评估

### 自测问卷（建议先做）

**Part A: Python 基础**

```python
# 问题 1：以下代码的输出是什么？
from pathlib import Path
p = Path("/home/user/project/model")
print(p / "config.json")

# 问题 2：这段代码在做什么？
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 问题 3：@decorator 是什么语法？
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function")
        return func(*args, **kwargs)
    return wrapper
```

**Part B: NumPy 基础**

```python
import numpy as np

# 问题 4：arr.shape 的值是什么？
arr = np.random.randn(3, 4, 5)

# 问题 5：result 的形状是什么？
a = np.random.randn(2, 3)
b = np.random.randn(2, 3)
result = a * b  # 注意：这是元素乘法

# 问题 6：这段代码在做什么？
mel = np.random.randn(128, 100)
mel_expanded = mel[np.newaxis, ...]  # (1, 128, 100)
```

**Part C: 深度学习基础**

```
# 问题 7：什么是 Epoch、Batch、Iteration？

# 问题 8：Forward 和 Backward 分别指什么？

# 问题 9：什么是 GPU 加速？为什么深度学习需要 GPU？
```

**Part D: 音频基础**

```
# 问题 10：16kHz 采样率意味着什么？

# 问题 11：40 秒的 16kHz 音频有多少个采样点？

# 问题 12：什么是频谱图（Spectrogram）？
```

### 自测结果评估

| 答对题数 | 建议路径 |
|---------|----------|
| 0-4 题 | 从**阶段一**开始系统学习 |
| 5-8 题 | 从**阶段二**开始，阶段一作为参考 |
| 9-12 题 | 直接进入**阶段三**，按需查阅 |

---

## 📖 阶段一：基础准备 (2-4 周)

### 🎯 阶段目标
掌握项目所需的最基础编程和数学知识。

---

### 第 1 步：Python 编程基础 (3-5 天)

#### 学习内容
| 主题 | 具体内容 | 重要性 |
|------|----------|--------|
| 函数定义 | 参数、返回值、默认值 | ⭐⭐⭐⭐⭐ |
| 类和对象 | `__init__`、方法、继承 | ⭐⭐⭐⭐⭐ |
| 模块导入 | `import`、`from ... import` | ⭐⭐⭐⭐⭐ |
| 文件操作 | `open()`、路径处理 | ⭐⭐⭐⭐ |
| 异常处理 | `try-except`、`raise` | ⭐⭐⭐⭐ |
| 类型注解 | `List`、`Dict`、`Optional` | ⭐⭐⭐ |

#### 学习资源
- 📖 《Python Crash Course》第 1-11 章
- 🌐 [Python 官方教程](https://docs.python.org/3/tutorial/)
- 📺 [B 站 Python 入门教程](https://www.bilibili.com/video/BV1ex411x7Em)

#### 实战练习
```python
# 练习 1.1：编写一个函数，统计文件夹中所有 .mp3 文件的数量
from pathlib import Path
from typing import List

def count_mp3_files(folder: str) -> int:
    """统计 MP3 文件数量"""
    # TODO: 实现此函数
    pass

# 练习 1.2：编写一个类，用于加载和预处理音频文件路径
class AudioLoader:
    def __init__(self, folder: str):
        self.folder = Path(folder)
    
    def get_all_audio_files(self, pattern: str = "*.mp3") -> List[Path]:
        """获取所有音频文件路径"""
        # TODO: 实现此函数
        pass
```

#### ✅ 检查点
- [ ] 能理解项目中的函数定义和调用
- [ ] 能理解类的使用（如 `QwenASREngine`）
- [ ] 能处理文件路径和读写操作

---

### 第 2 步：NumPy 数值计算 (3-5 天)

#### 学习内容
| 主题 | 具体内容 | 重要性 |
|------|----------|--------|
| 数组创建 | `np.array()`, `np.zeros()`, `np.random()` | ⭐⭐⭐⭐⭐ |
| 数组形状 | `shape`, `reshape()`, `transpose()` | ⭐⭐⭐⭐⭐ |
| 数组索引 | 切片、布尔索引、花式索引 | ⭐⭐⭐⭐⭐ |
| 数组运算 | 加减乘除、点积、广播机制 | ⭐⭐⭐⭐⭐ |
| 轴操作 | `axis` 参数、`sum()`, `mean()` | ⭐⭐⭐⭐ |

#### 学习资源
- 📖 《Python 深度学习》第 2 章
- 🌐 [NumPy 官方快速入门](https://numpy.org/doc/stable/user/quickstart.html)
- 📺 [NumPy 教程（B 站）](https://www.bilibili.com/video/BV1Z5411P7zE)

#### 实战练习
```python
import numpy as np

# 练习 2.1：理解音频数据的形状
# 创建一段 10 秒的模拟音频数据（16kHz 采样率）
sample_rate = 16000
duration = 10
audio = np.random.randn(sample_rate * duration).astype(np.float32)

print(f"音频形状：{audio.shape}")  # (160000,)
print(f"时长：{len(audio)/sample_rate} 秒")

# 练习 2.2：理解 Batch 维度
# 将单条音频添加 Batch 维度
audio_batch = audio[np.newaxis, :]  # (1, 160000)
print(f"Batch 形状：{audio_batch.shape}")

# 练习 2.3：理解 Mel 频谱的形状
# 128 维 Mel 频谱，100 帧
mel_spec = np.random.randn(128, 100)
print(f"Mel 频谱形状：{mel_spec.shape}")

# 添加 Batch 维度
mel_batch = mel_spec[np.newaxis, ...]  # (1, 128, 100)
print(f"Batch Mel 形状：{mel_batch.shape}")
```

#### ✅ 检查点
- [ ] 能理解 `(batch, freq, time)` 这样的多维数组
- [ ] 能理解 `np.newaxis` 和形状变换
- [ ] 能理解项目中的 NumPy 操作

---

### 第 3 步：音频基础概念 (2-3 天)

#### 学习内容
| 概念 | 说明 | 项目中的值 |
|------|------|-----------|
| 采样率 (Sample Rate) | 每秒采样次数 | 16000 Hz |
| 位深度 (Bit Depth) | 每个采样的精度 | 16/32-bit |
| 声道 (Channel) | 单声道/立体声 | 单声道 |
| 时域 vs 频域 | 波形图 vs 频谱图 | - |
| FFT | 快速傅里叶变换 | - |
| Mel 频谱 | 模拟人耳感知的频谱 | 128 维 |
| 分帧 (Framing) | 将音频切分为短帧 | 25ms/帧 |
| 跳帧 (Hop Length) | 帧与帧之间的间隔 | 160 采样点 |

#### 学习资源
- 📖 [数字信号处理基础](https://www.bilibili.com/video/BV1MK411X7bN)
- 🌐 [音频特征提取教程](https://www.kdnuggets.com/2020/02/audio-data-deep-learning.html)

#### 实战练习
```python
# 练习 3.1：计算音频参数
sample_rate = 16000
hop_length = 160
frame_size = 400  # 25ms

# 问题：40 秒音频有多少帧？
duration = 40
num_samples = sample_rate * duration
num_frames = num_samples // hop_length
print(f"40 秒音频的帧数：{num_frames}")  # 4000 帧

# 问题：每帧代表多少毫秒？
frame_duration_ms = frame_size / sample_rate * 1000
print(f"每帧时长：{frame_duration_ms} ms")  # 25ms
```

#### ✅ 检查点
- [ ] 理解采样率、帧、跳帧的概念
- [ ] 能计算音频时长和帧数的关系
- [ ] 理解 Mel 频谱的物理意义

---

### 阶段一总结

```
┌────────────────────────────────────────────────────┐
│  阶段一完成标准                                    │
├────────────────────────────────────────────────────┤
│  ✓ 能读懂项目中的 Python 代码结构                   │
│  ✓ 能理解 NumPy 数组的形状变换                     │
│  ✓ 能理解音频数据的基本表示方式                   │
│                                                    │
│  推荐下一步：进入阶段二（深度学习核心）            │
└────────────────────────────────────────────────────┘
```

---

## 📖 阶段二：核心知识 (4-6 周)

### 🎯 阶段目标
理解深度学习、Transformer、语音识别的核心概念。

---

### 第 4 步：深度学习基础 (1-2 周)

#### 学习内容
| 主题 | 具体内容 | 重要性 |
|------|----------|--------|
| 神经网络 | 神经元、层、激活函数 | ⭐⭐⭐⭐⭐ |
| 前向传播 | 输入→隐藏层→输出 | ⭐⭐⭐⭐⭐ |
| 损失函数 | MSE、CrossEntropy | ⭐⭐⭐⭐ |
| 反向传播 | 梯度、链式法则 | ⭐⭐⭐⭐ |
| 优化器 | SGD、Adam | ⭐⭐⭐⭐ |
| 训练流程 | Epoch、Batch、Iteration | ⭐⭐⭐⭐⭐ |
| 过拟合 | Dropout、正则化 | ⭐⭐⭐ |

#### 核心概念图解
```
输入层      隐藏层 1      隐藏层 2      输出层
  ○  ─────→  ○  ─────→  ○  ─────→  ○
  ○  ─────→  ○  ─────→  ○  ─────→  ○
  ○  ─────→  ○  ─────→  ○
         权重 W1        权重 W2
```

#### 学习资源
- 📖 《深度学习入门：基于 Python 的理论与实现》
- 📺 [吴恩达深度学习课程（中文字幕）](https://www.bilibili.com/video/BV16E411f7Rv)
- 🌐 [3Blue1Brown 神经网络可视化](https://www.3blue1brown.com/topics/neural-networks)

#### 实战练习
```python
import torch
import torch.nn as nn

# 练习 4.1：理解简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 64)   # 输入 128 维，输出 64 维
        self.relu = nn.ReLU()               # 激活函数
        self.layer2 = nn.Linear(64, 10)     # 输出 10 类
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 创建模型
model = SimpleNet()

# 模拟输入（Batch=4, 特征=128）
x = torch.randn(4, 128)

# 前向传播
output = model(x)
print(f"输出形状：{output.shape}")  # (4, 10)
```

#### ✅ 检查点
- [ ] 理解神经网络的前向传播过程
- [ ] 理解损失函数和优化器的作用
- [ ] 能区分训练和推理的不同

---

### 第 5 步：PyTorch 框架 (1-2 周)

#### 学习内容
| 主题 | 具体内容 | 重要性 |
|------|----------|--------|
| Tensor | 创建、操作、设备转移 | ⭐⭐⭐⭐⭐ |
| nn.Module | 模型定义、参数管理 | ⭐⭐⭐⭐⭐ |
| 常用层 | Linear、Conv2d、LayerNorm | ⭐⭐⭐⭐⭐ |
| 模型保存 | `load_state_dict()` | ⭐⭐⭐⭐⭐ |
| 设备管理 | CPU/GPU、`.to(device)` | ⭐⭐⭐⭐ |
| 梯度 | `no_grad()`、`eval()` | ⭐⭐⭐⭐ |

#### 学习资源
- 📖 [PyTorch 官方教程](https://pytorch.org/tutorials/)
- 🌐 [PyTorch 深度学习入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

#### 实战练习
```python
import torch
import torch.nn as nn

# 练习 5.1：理解项目中的模型加载
# 模拟 Qwen3ASRModel 的加载过程
model_name = "Qwen3-ASR-1.7B"

# 方式 1：从预训练权重加载
# model = Qwen3ASRModel.from_pretrained(model_name)

# 方式 2：设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# 方式 3：设置为评估模式（推理时使用）
# model.eval()

# 方式 4：不计算梯度（推理时节省内存）
with torch.no_grad():
    # output = model(input)
    pass

# 练习 5.2：理解常见的 PyTorch 操作
# 1. Tensor 形状变换
x = torch.randn(1, 128, 100)  # (Batch, Freq, Time)
x = x.transpose(1, 2)          # (Batch, Time, Freq)

# 2. 拼接操作
a = torch.randn(1, 50, 896)
b = torch.randn(1, 50, 896)
c = torch.cat([a, b], dim=1)   # (1, 100, 896)

# 3. 矩阵乘法
W = torch.randn(896, 1024)
x = torch.randn(1, 50, 896)
y = torch.matmul(x, W)         # (1, 50, 1024)
```

#### ✅ 检查点
- [ ] 能理解项目中的 PyTorch 代码
- [ ] 能使用 `nn.Module` 定义简单模型
- [ ] 理解 `eval()` 和 `no_grad()` 的作用

---

### 第 6 步：Transformer 架构 (2-3 周)

#### 学习内容
| 主题 | 具体内容 | 重要性 |
|------|----------|--------|
| Self-Attention | QKV 计算、注意力分数 | ⭐⭐⭐⭐⭐ |
| Multi-Head | 多头注意力机制 | ⭐⭐⭐⭐⭐ |
| Position Encoding | 位置信息注入 | ⭐⭐⭐⭐ |
| LayerNorm | 层归一化 | ⭐⭐⭐⭐ |
| Residual Connection | 残差连接 | ⭐⭐⭐⭐ |
| Feed-Forward | MLP 层 | ⭐⭐⭐⭐ |
| KV Cache | 推理加速技术 | ⭐⭐⭐⭐ |

#### Transformer 结构图
```
┌─────────────────────────────────────────┐
│           Transformer Block             │
├─────────────────────────────────────────┤
│  Input                                  │
│    ↓                                    │
│  LayerNorm                              │
│    ↓                                    │
│  Multi-Head Self-Attention              │
│    ↓                                    │
│  + Residual Connection (Add)            │
│    ↓                                    │
│  LayerNorm                              │
│    ↓                                    │
│  Feed-Forward Network (MLP)             │
│    ↓                                    │
│  + Residual Connection (Add)            │
│    ↓                                    │
│  Output                                 │
└─────────────────────────────────────────┘
```

#### 学习资源
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（强烈推荐）
- 📺 [李宏毅 Transformer 讲解（B 站）](https://www.bilibili.com/video/BV1Zn4y1f77f)
- 📺 [Transformer 工作原理（3Blue1Brown 风格）](https://www.bilibili.com/video/BV1pu411o7BE)

#### 实战练习
```python
import torch
import torch.nn as nn

# 练习 6.1：理解 Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
    
    def forward(self, x):
        # x: (Batch, Seq_Len, Embed_Dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention, V)
        return output

# 练习 6.2：理解项目中的 Transformer 使用
# Qwen3-ASR 的 Encoder 包含多层 Transformer
# 每层处理音频特征序列

# 输入：(Batch, Time, Embed_Dim)
# 输出：(Batch, Time, Embed_Dim)
```

#### ✅ 检查点
- [ ] 理解 Self-Attention 的计算过程
- [ ] 理解 Multi-Head 的作用
- [ ] 理解 Transformer 的输入输出形状

---

### 第 7 步：语音识别基础 (1-2 周)

#### 学习内容
| 主题 | 具体内容 | 重要性 |
|------|----------|--------|
| ASR 流程 | 音频→特征→编码→解码→文本 | ⭐⭐⭐⭐⭐ |
| 声学模型 | 音频→音素/字符 | ⭐⭐⭐⭐ |
| 语言模型 | 文本概率建模 | ⭐⭐⭐⭐ |
| CTC 算法 | 序列对齐 | ⭐⭐⭐ |
| Encoder-Decoder |  seq2seq 架构 | ⭐⭐⭐⭐⭐ |
| Whisper 架构 | 类似 Qwen3-ASR | ⭐⭐⭐⭐ |

#### Qwen3-ASR 架构图
```
┌──────────────────────────────────────────────────────┐
│              Qwen3-ASR 完整流程                       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  音频输入 (16kHz, 单声道)                            │
│     ↓                                                │
│  ┌────────────────────────────────────┐             │
│  │ Mel 频谱提取                        │             │
│  │ - FFT: 400 点                       │             │
│  │ - Hop: 160 采样点                   │             │
│  │ - Mel 滤波：128 维                   │             │
│  └────────────────────────────────────┘             │
│     ↓                                                │
│  ┌────────────────────────────────────┐             │
│  │ Audio Encoder                      │             │
│  │ - Frontend: CNN (分片处理)          │             │
│  │ - Backend: Transformer (24 层)       │             │
│  └────────────────────────────────────┘             │
│     ↓                                                │
│  音频 Embedding (T, 896)                             │
│     ↓                                                │
│  ┌────────────────────────────────────┐             │
│  │ LLM Decoder (Qwen3 1.7B)            │             │
│  │ - 输入：<audio> + Embedding         │             │
│  │ - 输出：转录文本                    │             │
│  └────────────────────────────────────┘             │
│     ↓                                                │
│  转录文本输出                                        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

#### 学习资源
- 📖 [语音识别入门教程](https://www.bilibili.com/video/BV1LE411X7bN)
- 📖 [Whisper 论文解读](https://www.bilibili.com/video/BV1wv4y1g7nW)
- 🌐 [HuggingFace ASR 课程](https://huggingface.co/learn/audio-course)

#### 实战练习
```python
# 练习 7.1：理解项目中的音频处理流程
from qwen_asr_gguf.inference.utils import load_audio
from qwen_asr_gguf.inference.encoder import QwenAudioEncoder

# 1. 加载音频
audio = load_audio("test.mp3")
print(f"音频形状：{audio.shape}")  # (N_samples,)

# 2. 编码为特征
encoder = QwenAudioEncoder("frontend.onnx", "backend.onnx")
embedding, time_cost = encoder.encode(audio)
print(f"Embedding 形状：{embedding.shape}")  # (T, 896)

# 练习 7.2：理解流式处理
# 40 秒音频被切分为多个片段处理
chunk_size = 40.0  # 秒
sample_rate = 16000
samples_per_chunk = int(chunk_size * sample_rate)  # 640000 采样点

# 计算需要多少个片段
num_chunks = len(audio) // samples_per_chunk + 1
print(f"需要处理 {num_chunks} 个片段")
```

#### ✅ 检查点
- [ ] 理解 ASR 的完整流程
- [ ] 理解 Encoder 和 Decoder 的作用
- [ ] 能解释项目中音频处理的每个步骤

---

### 阶段二总结

```
┌────────────────────────────────────────────────────┐
│  阶段二完成标准                                    │
├────────────────────────────────────────────────────┤
│  ✓ 理解深度学习的基本原理                          │
│  ✓ 能读懂 PyTorch 模型代码                          │
│  ✓ 理解 Transformer 的工作原理                      │
│  ✓ 理解语音识别的完整流程                          │
│                                                    │
│  推荐下一步：进入阶段三（进阶主题）                │
└────────────────────────────────────────────────────┘
```

---

## 📖 阶段三：进阶主题 (4-8 周)

### 🎯 阶段目标
掌握模型量化、ONNX、GGUF 等工程化技术。

---

### 第 8 步：模型量化 (1-2 周)

#### 学习内容
| 量化格式 | 说明 | 精度 | 大小 |
|---------|------|------|------|
| FP32 | 原始精度 | 100% | 4 bytes/参数 |
| FP16 | 半精度 | ~99.9% | 2 bytes/参数 |
| INT8 | 8 位整数 | ~99.5% | 1 byte/参数 |
| INT4 | 4 位整数 | ~98% | 0.5 bytes/参数 |
| Q4_K | K-quants | ~99% | 0.55 bytes/参数 |

#### 量化对比（以 1.7B 模型为例）
```
FP32:  1.7B × 4 bytes = 6.8 GB
FP16:  1.7B × 2 bytes = 3.4 GB
INT4:  1.7B × 0.5 bytes = 0.85 GB
Q4_K:  1.7B × 0.55 bytes = 0.94 GB

显存节省：8 倍！
速度提升：2-4 倍！
```

#### 学习资源
- 📖 [模型量化入门](https://zhuanlan.zhihu.com/p/639840696)
- 🌐 [GGUF 格式说明](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

#### ✅ 检查点
- [ ] 理解量化的原理和好处
- [ ] 能解释 FP16/INT4/Q4_K 的区别
- [ ] 理解项目中量化文件的使用

---

### 第 9 步：ONNX 格式 (1-2 周)

#### 学习内容
| 主题 | 具体内容 |
|------|----------|
| ONNX 模型结构 | 输入、输出、节点、图 |
| 模型导出 | `torch.onnx.export()` |
| ONNX Runtime | 推理引擎、执行提供程序 |
| 模型优化 | 算子融合、常量折叠 |

#### 项目中的 ONNX 使用
```python
import onnxruntime as ort

# 1. 加载模型
session = ort.InferenceSession("encoder.onnx")

# 2. 准备输入
input_data = np.random.randn(1, 128, 100).astype(np.float32)

# 3. 执行推理
output = session.run(None, {"chunk_mel": input_data})

# 4. 使用 GPU 加速（可选）
sess_opts = ort.SessionOptions()
providers = ['CPUExecutionProvider']
if use_dml:
    providers.insert(0, 'DmlExecutionProvider')
```

#### 学习资源
- 📖 [ONNX 官方教程](https://onnxruntime.ai/docs/tutorials/)
- 🌐 [PyTorch ONNX 导出](https://pytorch.org/tutorials/advanced/onnx.html)

#### ✅ 检查点
- [ ] 理解 ONNX 模型的结构
- [ ] 能使用 ONNX Runtime 进行推理
- [ ] 理解项目中 Encoder 的 ONNX 导出流程

---

### 第 10 步：GGUF 与 llama.cpp (2-4 周)

#### 学习内容
| 主题 | 具体内容 |
|------|----------|
| GGUF 格式 | 文件结构、元数据、张量 |
| llama.cpp | C++ 推理引擎 |
| Python 绑定 | ctypes 调用 DLL |
| GPU 加速 | Vulkan、DirectML、Metal |
| KV Cache | 推理加速技术 |

#### 项目中的 llama.cpp 使用
```python
from qwen_asr_gguf.inference import llama

# 1. 加载 GGUF 模型
model = llama.LlamaModel("qwen3_asr_llm.q4_k.gguf")

# 2. 创建上下文
ctx = llama.LlamaContext(model, n_ctx=2048)

# 3. 准备 Batch
batch = llama.LlamaBatch(max_tokens, embd_dim, 1)
batch.set_embd(embeddings, pos=positions)

# 4. 执行推理
ctx.decode(batch)

# 5. 获取 Logits
logits = ctx.get_logits()
```

#### 学习资源
- 📖 [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- 📖 [GGUF 格式详解](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

#### ✅ 检查点
- [ ] 理解 GGUF 文件的结构
- [ ] 能使用 llama.cpp 进行推理
- [ ] 理解项目中 Decoder 的 GGUF 加载流程

---

### 第 11 步：性能优化 (1-2 周)

#### 优化技术
| 技术 | 说明 | 效果 |
|------|------|------|
| 多进程并行 | 编码、解码、对齐异步 | 2-3x |
| Split ONNX | Encoder 分前后端 | 减少内存 |
| KV Cache | 复用历史 KV | 2x 推理速度 |
| 算子融合 | 合并多个算子 | 1.5x |
| GPU 加速 | DML/Vulkan | 5-10x |

#### 项目中的优化实践
```python
# 1. 多进程架构
# 主进程：控制流程 + Decoder
# 辅助进程：Encoder + Aligner

# 2. 流式处理
# 三级流水线：i+1 预取，i 识别，i-1 对齐

# 3. 记忆机制
# 保留历史片段的 Embedding 和文本
```

#### ✅ 检查点
- [ ] 理解多进程并行的好处
- [ ] 理解流式处理的原理
- [ ] 能分析性能瓶颈

---

### 阶段三总结

```
┌────────────────────────────────────────────────────┐
│  阶段三完成标准                                    │
├────────────────────────────────────────────────────┤
│  ✓ 理解模型量化的原理和应用                        │
│  ✓ 能使用 ONNX Runtime 进行推理                     │
│  ✓ 理解 GGUF 格式与 llama.cpp 的使用                 │
│  ✓ 能分析和优化性能瓶颈                            │
│                                                    │
│  恭喜！你已具备理解和修改本项目的能力              │
│  推荐下一步：阅读源码解析文档                      │
└────────────────────────────────────────────────────┘
```

---

## 📋 完整学习路径总览

### 时间规划

```
┌────────────────────────────────────────────────────────────────┐
│  学习路径时间轴（总计 10-18 周）                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  阶段一：基础准备 (2-4 周)                                       │
│  ├── 第 1 步：Python 基础 (3-5 天)                               │
│  ├── 第 2 步：NumPy 基础 (3-5 天)                               │
│  └── 第 3 步：音频基础 (2-3 天)                                 │
│                                                                │
│  阶段二：核心知识 (4-6 周)                                       │
│  ├── 第 4 步：深度学习基础 (1-2 周)                             │
│  ├── 第 5 步：PyTorch 框架 (1-2 周)                             │
│  ├── 第 6 步：Transformer (2-3 周)                              │
│  └── 第 7 步：语音识别基础 (1-2 周)                             │
│                                                                │
│  阶段三：进阶主题 (4-8 周)                                       │
│  ├── 第 8 步：模型量化 (1-2 周)                                 │
│  ├── 第 9 步：ONNX 格式 (1-2 周)                                │
│  ├── 第 10 步：GGUF 与 llama.cpp (2-4 周)                        │
│  └── 第 11 步：性能优化 (1-2 周)                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 快速路径 vs 完整路径

| 路径 | 适合人群 | 预计时间 | 学习内容 |
|------|----------|----------|----------|
| **快速路径** | 有基础，想快速使用 | 1-2 周 | 阶段一 + 阶段二第 7 步 |
| **标准路径** | 想深入理解 | 8-12 周 | 阶段一 + 阶段二 + 阶段三部分 |
| **完整路径** | 想修改/扩展 | 12-18 周 | 全部内容 |

---

## 🎯 实战项目建议

### 项目 1：音频转录工具（阶段一后）
```python
# 目标：使用现有 API 完成转录
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig

def transcribe_audio(audio_path: str) -> str:
    config = ASREngineConfig(model_dir="model")
    engine = QwenASREngine(config)
    result = engine.transcribe(audio_path)
    engine.shutdown()
    return result.text
```

### 项目 2：批量处理脚本（阶段二后）
```python
# 目标：批量处理文件夹中的音频
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def batch_transcribe(folder: str):
    audio_files = list(Path(folder).glob("*.mp3"))
    # 实现批量处理逻辑
```

### 项目 3：Web 服务（阶段三后）
```python
# 目标：部署为 Web 服务
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 实现 Web API
```

---

## 📚 推荐资源汇总

### 书籍
| 书名 | 难度 | 适合阶段 |
|------|------|----------|
| 《Python Crash Course》 | 入门 | 阶段一 |
| 《深度学习入门》 | 入门 | 阶段二 |
| 《Python 深度学习》 | 进阶 | 阶段二 |
| 《语音识别实战》 | 进阶 | 阶段二 |

### 在线课程
| 课程 | 平台 | 适合阶段 |
|------|------|----------|
| 吴恩达深度学习 | Coursera | 阶段二 |
| PyTorch 官方教程 | PyTorch.org | 阶段二 |
| HuggingFace 音频课程 | HuggingFace | 阶段二 |

### 视频资源
| 内容 | B 站链接 | 适合阶段 |
|------|----------|----------|
| Python 入门 | BV1ex411x7Em | 阶段一 |
| 吴恩达深度学习 | BV16E411f7Rv | 阶段二 |
| 李宏毅 Transformer | BV1Zn4y1f77f | 阶段二 |
| 音频信号处理 | BV1MK411X7bN | 阶段一 |

---

## ❓ 常见问题

### Q1: 我需要数学基础吗？
**A:** 基础线性代数（矩阵运算）和微积分（导数）概念即可。深入学习时需要，但使用层面影响不大。

### Q2: 需要 GPU 吗？
**A:** 
- 学习 Python/NumPy：不需要
- 学习 PyTorch：可选（CPU 也能学）
- 运行本项目：推荐有 GPU（DML/Vulkan）

### Q3: 学完能做什么？
**A:** 
- 使用 Qwen3-ASR 进行语音转录
- 转换和量化自己的模型
- 修改代码适配新场景
- 开发类似的语音应用

### Q4: 学习过程中遇到问题怎么办？
**A:** 
1. 先查阅项目文档
2. 搜索相关技术文档
3. 查看项目 Issues
4. 在社区提问

---

## 📝 学习进度追踪

### 自我评估表

| 阶段 | 完成日期 | 掌握程度 | 备注 |
|------|----------|----------|------|
| 阶段一 - Python | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段一 - NumPy | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段一 - 音频 | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段二 - 深度学习 | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段二 - PyTorch | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段二 - Transformer | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段二 - 语音识别 | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段三 - 量化 | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段三 - ONNX | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段三 - GGUF | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |
| 阶段三 - 优化 | ____/__/__ | □ 未开始 □ 学习中 □ 已完成 | |

---

## 🎓 下一步建议

完成本学习计划后，建议：

1. **阅读架构文档** (`02-ARCHITECTURE.md`) - 深入理解项目设计
2. **阅读源码解析** (`SOURCE_CODE.md`) - 逐模块分析代码
3. **阅读导出指南** (`EXPORT_GUIDE.md`) - 学习模型转换
4. **动手实践** - 尝试修改和扩展项目

---

**文档结束**

---

## 附录：文档阅读顺序总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen3-ASR-GGUF 文档体系                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① 02-ARCHITECTURE.md（架构文档）                                  │
│     └── 目标：了解项目整体设计和技术选型                         │
│     └── 适合：所有人，1-2 小时阅读                               │
│                                                                 │
│  ② 03-INTEGRATION.md（集成指南）                                   │
│     └── 目标：快速上手使用项目                                   │
│     └── 适合：想快速使用的开发者，1-2 小时阅读                    │
│                                                                 │
│  ③ 06-LEARNING_PLAN.md（本学习计划）                               │
│     └── 目标：系统学习所需知识                                   │
│     └── 适合：想深入理解原理的开发者，4-12 周学习                 │
│                                                                 │
│  ④ EXPORT_GUIDE.md（导出指南）                                  │
│     └── 目标：学习模型转换流程                                   │
│     └── 适合：想转换自己模型的开发者，2-4 小时阅读                │
│                                                                 │
│  ⑤ SOURCE_CODE.md（源码解析）                                   │
│     └── 目标：深入理解代码实现                                   │
│     └── 适合：想修改扩展项目的开发者，4-8 周研读                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
