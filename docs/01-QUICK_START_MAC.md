# macOS 小白快速上手指南

> 目标：在 macOS 上，从零到成功转录音频，约 10 分钟
> 最后更新：2026-02-24

---

## 前置说明

### 两条推理路径

本项目有两条独立的推理路径：

| 路径 | 特点 | 适合场景 |
|------|------|----------|
| **GGUF 路径**（推荐） | 速度快，RTF ≈ 0.1，无需 GPU | 日常使用 |
| PyTorch 路径 | 依赖重（torch 1.8GB+），RTF ≈ 0.4 | 研究/调试 |

**本指南只讲 GGUF 路径**，这也是项目的主推方案。

### 为什么 Mac 不需要编译 llama.cpp？

GGUF 推理依赖 `libllama.dylib` 等动态库。有人可能想到用 Ollama——但 Ollama 把 llama.cpp **静态编译**进了主二进制，没有暴露 `.dylib`，无法复用。

好消息是：**本项目已将编译好的 macOS dylibs 提交到仓库**（`qwen_asr_gguf/inference/bin/`），克隆仓库即自动获得，**不需要自己编译**。

---

## 第 1 步：克隆仓库

```bash
git clone https://github.com/HaujetZhao/Qwen3-ASR-GGUF.git
cd Qwen3-ASR-GGUF
```

克隆后 `qwen_asr_gguf/inference/bin/` 已包含以下 dylibs（共约 4.4MB）：

```
libllama.dylib        ← llama.cpp 核心（GGUF 解码，Metal 加速）
libggml.dylib
libggml-base.dylib
libggml-cpu.dylib
libggml-metal.dylib   ← Apple Metal GPU 加速
libggml-blas.dylib
```

---

## 第 2 步：安装依赖

### 2.1 安装 ffmpeg（音频解码支持）

```bash
brew install ffmpeg
```

> **为什么需要 ffmpeg？** 项目用 `pydub` 加载音频，`pydub` 内部调用 `ffmpeg` 解码 m4a、mp3、webm 等格式。

如果还没有 Homebrew：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2.2 安装 Python 依赖

```bash
pip install onnxruntime numpy scipy gguf pydub srt typer rich
```

**Mac 上不要安装 `onnxruntime-directml`**（DirectML 是 Windows 专属的 GPU API，在 Mac 上无法安装）。`requirements.txt` 里有这一行，在 Mac 上直接忽略即可，安装上面的 `onnxruntime` 替代。

> **各包用途速查：**
> - `onnxruntime` — 运行 ONNX 格式的音频编码器（Encoder）
> - `numpy` / `scipy` — Mel 频谱提取（信号处理）
> - `gguf` — 读取 GGUF 文件的 token 嵌入表
> - `pydub` — 加载各格式音频文件
> - `srt` — 导出 SRT 字幕文件
> - `typer` / `rich` — `transcribe.py` 的命令行界面

---

## 第 3 步：下载模型文件

从 GitHub Releases 下载模型包（选择合适的大小）：

| 模型 | 文件名 | 大小 | 说明 |
|------|--------|------|------|
| **0.6B（推荐）** | `Qwen3-ASR-0.6B-gguf.zip` | 538 MB | 速度快，精度够用 |
| 1.7B | `Qwen3-ASR-1.7B-gguf.zip` | 1.3 GB | 精度更高，速度稍慢 |

**下载方式一：命令行（需要 GitHub CLI）**

```bash
# 安装 gh（如果还没有）
brew install gh

# 下载并解压到 model/ 目录
gh release download models \
  --repo HaujetZhao/Qwen3-ASR-GGUF \
  --pattern "Qwen3-ASR-0.6B-gguf.zip" \
  --dir /tmp/

unzip /tmp/Qwen3-ASR-0.6B-gguf.zip -d model/
```

**下载方式二：浏览器手动下载**

访问 https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models，下载 zip 文件后解压到项目根目录下的 `model/` 文件夹（没有就创建）。

解压后 `model/` 目录应包含：

```
model/
  qwen3_asr_encoder_frontend.int4.onnx   ← 编码器前端（~19 MB）
  qwen3_asr_encoder_backend.int4.onnx    ← 编码器后端（~90 MB）
  qwen3_asr_llm.q4_k.gguf               ← 解码器 LLM（~462 MB）
```

> **注意**：务必前台运行下载命令（不要加 `&` 放后台），否则可能下载不完整导致解压失败。

---

## 第 4 步：转录音频

```bash
python transcribe.py 你的音频.m4a --no-dml
```

`--no-dml` 表示关闭 DirectML（Mac 上没有这个加速方式，但默认值是 ON，需要手动关掉）。

### 常用选项

```bash
# 指定语言（加快速度）
python transcribe.py audio.m4a --no-dml --language Chinese

# 添加上下文提示（提高专有名词识别率）
python transcribe.py audio.m4a --no-dml --context "这是一段关于机器学习的讲座"

# 安静模式（不打印详细日志）
python transcribe.py audio.m4a --no-dml --quiet

# 覆盖已有输出文件
python transcribe.py audio.m4a --no-dml -y
```

### 输出文件

转录完成后，在音频文件同目录下会生成：

```
audio.txt   ← 纯文本（每行一句）
audio.srt   ← SRT 字幕（含时间戳，需开启时间戳对齐）
audio.json  ← JSON（含详细时间戳，需开启时间戳对齐）
```

---

## 加速说明（Mac 专项）

配置面板中显示 `DML:OFF | Vulkan:OFF`，这两项都是 Windows/Linux 加速方案，**在 Mac 上正常就是 OFF**，不代表没有加速。

Mac 实际上有两层加速在工作：

| 组件 | 加速方式 | 状态 |
|------|----------|------|
| **GGUF 解码器**（LLM） | Apple Metal GPU | 自动启用（`libggml-metal.dylib`） |
| **ONNX 编码器** | CPU（AVX2 优化） | 运行中 |

实测性能（M 系芯片 Mac，0.6B 模型，16 秒音频）：

| 指标 | 数值 |
|------|------|
| RTF（实时率） | 约 0.10 |
| 含义 | 转录 1 分钟音频仅需约 6 秒 |

---

## 常见错误速查

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `No such file: model/qwen3_asr_llm.q4_k.gguf` | 模型文件未下载 | 执行第 3 步下载模型 |
| `ModuleNotFoundError: No module named 'onnxruntime'` | 依赖未安装 | `pip install onnxruntime` |
| `pydub.exceptions.CouldntDecodeAudio` | ffmpeg 未安装 | `brew install ffmpeg` |
| `End-of-central-directory signature not found` | 下载的 zip 不完整 | 重新前台下载（不要加 `&`） |
| `ERROR: Could not find a version that satisfies onnxruntime-directml` | 装了 Windows 包 | 改装 `pip install onnxruntime` |
| 程序启动后立即崩溃（segfault） | 自带的 dylib 版本不对 | 确认 `bin/` 中的是仓库自带的（不要替换为 llama-cpp-python 的） |

---

## 完整安装命令汇总

```bash
# 1. 克隆仓库
git clone https://github.com/HaujetZhao/Qwen3-ASR-GGUF.git
cd Qwen3-ASR-GGUF

# 2. 系统依赖
brew install ffmpeg

# 3. Python 依赖
pip install onnxruntime numpy scipy gguf pydub srt typer rich

# 4. 下载模型（选其一）
# 方式一：命令行
gh release download models \
  --repo HaujetZhao/Qwen3-ASR-GGUF \
  --pattern "Qwen3-ASR-0.6B-gguf.zip" \
  --dir /tmp/
unzip /tmp/Qwen3-ASR-0.6B-gguf.zip -d model/

# 方式二：浏览器下载后手动移动
mkdir -p model && mv ~/Downloads/Qwen3-ASR-0.6B-gguf/* model/

# 5. 运行转录
python transcribe.py 你的音频.m4a --no-dml
```
