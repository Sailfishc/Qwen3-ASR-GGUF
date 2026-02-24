# Qwen3-ASR-GGUF 集成指南

> 文档版本：1.0  
> 最后更新：2026-02-23  
> **推荐阅读顺序：第 ② 顺位**（架构理解后的实战指南）

---

## 📋 本文档阅读指南

### 在完整项目文档中的阅读顺序

| 顺位 | 文档 | 文件名 | 目标读者 | 预计耗时 |
|:----:|------|--------|----------|----------|
| **①** | [项目架构](./02-ARCHITECTURE.md) | `02-ARCHITECTURE.md` | 想了解项目整体设计 | 1-2 小时 |
| **②** | **集成指南** | `03-INTEGRATION.md` | 想快速使用项目 | 1-2 小时 |
| **③** | [学习计划](./06-LEARNING_PLAN.md) | `06-LEARNING_PLAN.md` | 想深入理解原理 | 4-12 周 |
| **④** | [导出指南](./EXPORT_GUIDE.md) | `EXPORT_GUIDE.md` | 想转换自己的模型 | 2-4 小时 |
| **⑤** | [源码解析](./SOURCE_CODE.md) | `SOURCE_CODE.md` | 想修改/扩展功能 | 4-8 周 |

### 本文档结构

```
阅读建议：根据你的需求选择阅读路径

快速使用路径：
  1. 快速开始 ──────▶ 5 分钟上手
  2. 安装与配置 ────▶ 环境搭建
  3. Python 集成 ───▶ 代码调用

深入学习路径：
  1. 快速开始 ──────▶ 了解基本用法
  2. Python 集成 ───▶ 完整 API 使用
  3. 高级配置 ──────▶ 性能调优
  4. 错误处理 ─────▶ 问题排查

部署应用路径：
  1. Web 服务集成 ──▶ FastAPI 部署
  2. 批量处理 ─────▶ 大规模处理
  3. Docker 部署 ───▶ 容器化部署
```

---

## 目录

1. [快速开始](#1-快速开始)
2. [安装与配置](#2-安装与配置)
3. [Python 集成](#3-python 集成)
4. [命令行集成](#4-命令行集成)
5. [Web 服务集成](#5-Web 服务集成)
6. [批量处理](#6-批量处理)
7. [高级配置](#7-高级配置)
8. [错误处理与调试](#8-错误处理与调试)
9. [性能优化](#9-性能优化)
10. [常见问题](#10-常见问题)

---

## 1. 快速开始

### 1.1 最小可用示例

```python
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig

# 配置引擎
config = ASREngineConfig(model_dir="model")

# 初始化引擎
engine = QwenASREngine(config)

# 执行转录
result = engine.transcribe("audio.mp3")

# 输出结果
print(result.text)

# 关闭引擎
engine.shutdown()
```

### 1.2 一行命令转录

```bash
python transcribe.py audio.mp3 -y
```

---

## 2. 安装与配置

### 2.1 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.8+ | 3.10+ |
| 操作系统 | Windows 10 / macOS 10.15 / Linux | - |
| 显存 | 4GB (CPU 模式) | 8GB+ (GPU 模式) |

### 2.2 安装依赖

#### 基础依赖

```bash
pip install -r requirements.txt
```

核心依赖说明：

```txt
# 模型转换
transformers==4.57.6
torch
accelerate

# 推理引擎
onnxruntime-directml    # Windows DirectML
# 或
onnxruntime-gpu         # Linux/Mac CUDA

gguf                    # GGUF 格式支持

# 音频处理
pydub                   # 音频加载
librosa                 # 音频特征 (可选)
srt                     # 字幕生成

# 分词支持 (可选)
nagisa                  # 日文分词
```

#### llama.cpp 动态库

从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载预编译二进制：

| 平台 | 下载文件 | 解压后位置 |
|------|----------|------------|
| **Windows (DML)** | `llama-bXXXX-bin-win-dml-x64.zip` | `qwen_asr_gguf/inference/bin/` |
| **Windows (Vulkan)** | `llama-bXXXX-bin-win-vulkan-x64.zip` | `qwen_asr_gguf/inference/bin/` |
| **macOS** | `llama-bXXXX-bin-macos-x64.zip` | `qwen_asr_gguf/inference/bin/` |
| **Linux** | 需从源码编译 | `qwen_asr_gguf/inference/bin/` |

所需 DLL 文件：

```
qwen_asr_gguf/inference/bin/
├── ggml.dll (或 libggml.so / libggml.dylib)
├── ggml-base.dll
└── llama.dll (或 libllama.so / libllama.dylib)
```

### 2.3 下载模型

#### 方式 1：下载预转换模型（推荐）

从 [GitHub Releases](https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models) 下载已转换好的模型：

```bash
# 下载后解压到项目根目录的 model 文件夹
# 推荐下载量化版本，节省显存
```

#### 方式 2：手动转换模型

```bash
# 1. 下载原始模型
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B

# 2. 配置路径 (export_config.py)
from pathlib import Path
model_home = Path('~/.cache/modelscope/hub/models/Qwen').expanduser()

ASR_MODEL_DIR = model_home / 'Qwen3-ASR-1.7B'
ALIGNER_MODEL_DIR = model_home / 'Qwen3-ForcedAligner-0.6B'
EXPORT_DIR = r'./model'

# 3. 执行转换
python 01-Export-ASR-Encoder-Frontend.py
python 02-Export_ASR-Encoder-Backend.py
python 03-Optimize-ASR-Encoder.py
python 04-Quantize-ASR-Encoder.py
python 05-Export-ASR-Decoder-HF.py
python 06-Convert-ASR-Decoder-GGUF.py
python 07-Quantize-ASR-Decoder-GGUF.py

# Aligner 模型 (可选)
python 11-Export-Aligner-Encoder-Frontend.py
# ... (12-17 步骤相同)
```

### 2.4 验证安装

```python
# test_install.py
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig
from pathlib import Path

# 检查模型文件
model_files = [
    "model/qwen3_asr_llm.q4_k.gguf",
    "model/qwen3_asr_encoder_frontend.int4.onnx",
    "model/qwen3_asr_encoder_backend.int4.onnx"
]

for f in model_files:
    if not Path(f).exists():
        print(f"❌ 缺失模型文件：{f}")
    else:
        print(f"✅ 模型文件存在：{f}")

# 测试初始化
try:
    config = ASREngineConfig(model_dir="model")
    engine = QwenASREngine(config)
    print("✅ 引擎初始化成功")
    engine.shutdown()
except Exception as e:
    print(f"❌ 引擎初始化失败：{e}")
```

---

## 3. Python 集成

### 3.1 基础集成

#### 3.1.1 简单转录

```python
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig

def transcribe_audio(audio_path: str):
    """最简单的转录函数"""
    config = ASREngineConfig(model_dir="model")
    engine = QwenASREngine(config)
    
    try:
        result = engine.transcribe(audio_path)
        return result.text
    finally:
        engine.shutdown()

# 使用示例
text = transcribe_audio("meeting.mp3")
print(text)
```

#### 3.1.2 带配置的转录

```python
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, AlignerConfig

# 完整配置
config = ASREngineConfig(
    model_dir="/path/to/models",          # 模型目录
    encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
    encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
    llm_fn="qwen3_asr_llm.q4_k.gguf",
    
    # 硬件加速
    use_dml=True,              # Windows DirectML
    n_ctx=2048,                # 上下文窗口
    
    # 流式配置
    chunk_size=40.0,           # 每片 40 秒
    memory_num=1,              # 记忆 1 个历史片段
    
    # 对齐配置
    enable_aligner=True,
    align_config=AlignerConfig(
        use_dml=True,
        model_dir="/path/to/models"
    ),
    
    verbose=True
)

engine = QwenASREngine(config)
result = engine.transcribe(
    audio_file="audio.mp3",
    context="会议录音，包含技术讨论",  # 上下文提示
    language="Chinese",              # 强制语言
    temperature=0.4                  # 采样温度
)

print(f"转录文本：{result.text}")
print(f"性能统计：{result.performance}")

if result.alignment:
    print(f"字级时间戳：{len(result.alignment.items)} 个")

engine.shutdown()
```

### 3.2 结果处理

#### 3.2.1 导出结果

```python
from qwen_asr_gguf.inference import exporters

# 导出 TXT
exporters.export_to_txt("output.txt", result)

# 导出 SRT 字幕
exporters.export_to_srt("output.srt", result)

# 导出 JSON 时间戳
exporters.export_to_json("output.json", result)
```

#### 3.2.2 自定义后处理

```python
from qwen_asr_gguf.inference import chinese_itn

# 中文数字规整
text = result.text
normalized = chinese_itn.chinese_to_num(text)

# 示例：'二零二五年' → '2025 年'
#      '一百二十三人' → '123 人'
```

#### 3.2.3 访问对齐结果

```python
if result.alignment:
    for item in result.alignment.items[:10]:  # 前 10 个字
        print(f"{item.text}: {item.start_time:.3f}s - {item.end_time:.3f}s")
    
    # 转换为字典列表
    items_dict = [
        {
            "text": item.text,
            "start": item.start_time,
            "end": item.end_time
        }
        for item in result.alignment.items
    ]
```

### 3.3 音频切片处理

#### 3.3.1 指定时间范围

```python
# 从第 30 秒开始，读取 60 秒
result = engine.transcribe(
    audio_file="long_audio.mp3",
    start_second=30.0,
    duration=60.0
)
```

#### 3.3.2 分段处理长音频

```python
def process_long_audio(engine, audio_path, chunk_minutes=10):
    """处理超长音频，分段转录"""
    from qwen_asr_gguf.inference.utils import load_audio
    
    # 加载音频获取总时长
    audio = load_audio(audio_path)
    total_duration = len(audio) / 16000  # 秒
    
    results = []
    chunk_seconds = chunk_minutes * 60
    
    for start in range(0, int(total_duration), chunk_seconds):
        duration = min(chunk_seconds, total_duration - start)
        
        result = engine.transcribe(
            audio_file=audio_path,
            start_second=start,
            duration=duration,
            context=f"第 {start//60 + 1} 段"
        )
        
        results.append({
            "start": start,
            "duration": duration,
            "text": result.text
        })
        
        print(f"完成片段：{start}s - {start+duration}s")
    
    return results

# 使用示例
segments = process_long_audio(engine, "lecture.mp3", chunk_minutes=10)
full_text = "\n".join([s["text"] for s in segments])
```

### 3.4 多语言支持

```python
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese",
    "Japanese", "Korean", "French", "German",
    "Spanish", "Russian", "Arabic", "Thai",
    "Vietnamese", "Indonesian", "Hindi"
    # ... 共 28 种语言
]

# 自动语言识别 (默认)
result = engine.transcribe(audio_path, language=None)

# 强制指定语言
result = engine.transcribe(audio_path, language="English")

# 中英混合 (推荐用 Chinese，模型可处理混合)
result = engine.transcribe(audio_path, language="Chinese")
```

---

## 4. 命令行集成

### 4.1 基础用法

```bash
# 基本转录
python transcribe.py audio.mp3

# 输出到指定文件
python transcribe.py audio.mp3 -m ./model --prec int4
```

### 4.2 完整参数说明

```bash
python transcribe.py audio.mp3 \
    # === 模型配置 ===
    --model-dir ./model \
    --prec int4 \              # fp32, fp16, int8, int4
    --timestamp / --no-ts \    # 时间戳对齐
    --dml / --no-dml \         # DirectML 加速
    --vulkan / --no-vulkan \   # Vulkan 加速
    --n-ctx 2048 \             # 上下文窗口
    
    # === 转录设置 ===
    --language Chinese \       # 强制语种
    --context "会议录音" \      # 上下文提示
    --temperature 0.4 \        # 采样温度
    
    # === 音频切片 ===
    --seek-start 30 \          # 开始位置 (秒)
    --duration 60 \            # 处理时长 (秒)
    
    # === 流式配置 ===
    --chunk-size 40 \          # 分段时长 (秒)
    --memory-num 1 \           # 记忆片段数
    
    # === 其他 ===
    --verbose / --quiet \      # 详细日志
    --yes                      # 覆盖已存在文件
```

### 4.3 批处理脚本

#### Windows Batch

```batch
@echo off
set MODEL_DIR=model
set OUTPUT_DIR=output

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

for %%f in (audio\*.mp3) do (
    echo 正在处理：%%f
    python transcribe.py "%%f" ^
        --model-dir %MODEL_DIR% ^
        --prec int4 ^
        --dml ^
        --yes
)

echo 批量处理完成
```

#### Linux/Mac Shell

```bash
#!/bin/bash

MODEL_DIR="model"
OUTPUT_DIR="output"

mkdir -p "$OUTPUT_DIR"

for file in audio/*.mp3; do
    echo "Processing: $file"
    python transcribe.py "$file" \
        --model-dir "$MODEL_DIR" \
        --prec int4 \
        --dml \
        --yes
done

echo "Batch processing complete"
```

#### Python 批量处理

```python
from pathlib import Path
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, exporters

def batch_transcribe(audio_folder: str, output_folder: str):
    """批量转录文件夹中的所有音频"""
    config = ASREngineConfig(model_dir="model")
    engine = QwenASREngine(config)
    
    try:
        audio_files = list(Path(audio_folder).glob("*.mp3"))
        audio_files.extend(Path(audio_folder).glob("*.wav"))
        audio_files.extend(Path(audio_folder).glob("*.m4a"))
        
        for audio_path in audio_files:
            print(f"\n处理：{audio_path.name}")
            
            result = engine.transcribe(str(audio_path))
            
            # 导出结果
            base_name = audio_path.stem
            exporters.export_to_txt(f"{output_folder}/{base_name}.txt", result)
            exporters.export_to_srt(f"{output_folder}/{base_name}.srt", result)
            exporters.export_to_json(f"{output_folder}/{base_name}.json", result)
            
    finally:
        engine.shutdown()

# 使用示例
batch_transcribe("audio_files", "transcriptions")
```

---

## 5. Web 服务集成

### 5.1 FastAPI 服务

```python
# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os

from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, exporters

app = FastAPI(title="Qwen3-ASR Service")

# 全局引擎 (单例)
engine = None

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化引擎"""
    global engine
    config = ASREngineConfig(
        model_dir="model",
        use_dml=True,
        enable_aligner=True
    )
    engine = QwenASREngine(config)
    print("ASR Engine initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时释放资源"""
    global engine
    if engine:
        engine.shutdown()

class TranscribeRequest(BaseModel):
    language: Optional[str] = None
    context: Optional[str] = None
    temperature: float = 0.4

class TranscribeResponse(BaseModel):
    text: str
    duration: float
    performance: dict
    srt_available: bool
    json_available: bool

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    context: Optional[str] = None,
    temperature: float = 0.4
):
    """转录上传的音频文件"""
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="不支持的音频格式")
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 执行转录
        result = engine.transcribe(
            audio_file=tmp_path,
            language=language,
            context=context,
            temperature=temperature
        )
        
        # 导出 SRT 和 JSON
        srt_path = tmp_path + ".srt"
        json_path = tmp_path + ".json"
        exporters.export_to_srt(srt_path, result)
        exporters.export_to_json(json_path, result)
        
        return TranscribeResponse(
            text=result.text,
            duration=result.performance.get('total_time', 0),
            performance=result.performance,
            srt_available=os.path.exists(srt_path),
            json_available=os.path.exists(json_path)
        )
        
    finally:
        # 清理临时文件
        for path in [tmp_path, tmp_path + ".srt", tmp_path + ".json"]:
            if os.path.exists(path):
                os.remove(path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 启动服务
# uvicorn server:app --host 0.0.0.0 --port 8000
```

### 5.2 客户端调用示例

```python
import requests

# 上传文件转录
with open("audio.mp3", "rb") as f:
    files = {"file": f}
    data = {
        "language": "Chinese",
        "context": "技术会议录音",
        "temperature": 0.4
    }
    
    response = requests.post(
        "http://localhost:8000/transcribe",
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"转录文本：{result['text']}")
    print(f"处理耗时：{result['duration']:.2f}秒")
```

### 5.3 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用
COPY . .

# 下载模型 (或使用挂载卷)
# RUN python download_models.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  asr-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model
      - ./uploads:/app/uploads
    environment:
      - PYTHONUNBUFFERED=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 6. 批量处理

### 6.1 多文件并行处理

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig

def transcribe_single_file(args):
    """单文件转录函数"""
    audio_path, config_data = args
    
    # 每个线程独立引擎实例
    config = ASREngineConfig(**config_data)
    engine = QwenASREngine(config)
    
    try:
        result = engine.transcribe(audio_path)
        return {
            "file": audio_path,
            "text": result.text,
            "success": True
        }
    except Exception as e:
        return {
            "file": audio_path,
            "error": str(e),
            "success": False
        }
    finally:
        engine.shutdown()

def batch_parallel(audio_files, max_workers=4):
    """并行批量处理"""
    config_data = {
        "model_dir": "model",
        "use_dml": True,
        "verbose": False
    }
    
    args_list = [(f, config_data) for f in audio_files]
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(transcribe_single_file, args): args[0] 
                   for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                print(f"✅ {result['file']}: {len(result['text'])} 字符")
            else:
                print(f"❌ {result['file']}: {result['error']}")
            results[result['file']] = result
    
    return results

# 使用示例
audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]
results = batch_parallel(audio_files, max_workers=2)
```

### 6.2 进度追踪

```python
from tqdm import tqdm
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, exporters

def batch_with_progress(audio_folder: str, output_folder: str):
    """带进度条的批量处理"""
    from pathlib import Path
    
    audio_files = list(Path(audio_folder).glob("*.mp3"))
    
    config = ASREngineConfig(model_dir="model", verbose=False)
    engine = QwenASREngine(config)
    
    try:
        for audio_path in tqdm(audio_files, desc="处理音频"):
            result = engine.transcribe(str(audio_path))
            
            base_name = audio_path.stem
            exporters.export_to_txt(f"{output_folder}/{base_name}.txt", result)
            
    finally:
        engine.shutdown()
```

---

## 7. 高级配置

### 7.1 硬件加速配置

#### DirectML (Windows)

```python
config = ASREngineConfig(
    model_dir="model",
    use_dml=True,  # 启用 DirectML
)

# 环境变量配置
import os
os.environ["GGML_DIRECTML_LOG"] = "0"  # 0=关闭日志
```

#### Vulkan (跨平台)

```python
config = ASREngineConfig(
    model_dir="model",
    use_dml=False,
)

# Vulkan 环境变量
import os
os.environ["GGML_VULKAN_DEVICE"] = "0"  # 选择 GPU 设备
os.environ["GGML_VULKAN_LOG"] = "0"

# Intel 集显 FP16 问题
os.environ["GGML_VULKAN_DISABLE_F16"] = "1"
```

#### CPU 模式

```python
config = ASREngineConfig(
    model_dir="model",
    use_dml=False,
    n_ctx=1024,  # 减小上下文节省内存
)
```

### 7.2 性能调优参数

```python
config = ASREngineConfig(
    model_dir="model",
    
    # 上下文窗口 (越大越占显存，但上下文更长)
    n_ctx=2048,
    
    # 流式切片
    chunk_size=40.0,    # 每片秒数 (默认 40s)
    memory_num=1,       # 记忆历史片段数 (0=无记忆)
    
    # 对齐引擎
    enable_aligner=True,
)
```

### 7.3 上下文提示优化

```python
# 场景 1：会议录音
context = "会议录音，包含多位发言人讨论技术方案"

# 场景 2：播客节目
context = "播客节目，主持人和嘉宾对话"

# 场景 3：专业领域
context = "医学讲座，包含专业术语"

# 场景 4：带专有名词
context = "产品发布会，提及 iPhone、MacBook、Apple Watch 等产品"

result = engine.transcribe(audio_path, context=context)
```

### 7.4 自定义导出格式

```python
from qwen_asr_gguf.inference.schema import TranscribeResult
import json

def export_custom_format(result: TranscribeResult, output_path: str):
    """自定义 JSON 导出"""
    data = {
        "full_text": result.text,
        "segments": [],
        "performance": result.performance
    }
    
    if result.alignment:
        data["word_level"] = [
            {
                "text": item.text,
                "start": round(item.start_time, 3),
                "end": round(item.end_time, 3)
            }
            for item in result.alignment.items
        ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

---

## 8. 错误处理与调试

### 8.1 常见错误处理

```python
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig
import os

def safe_transcribe(audio_path: str):
    """带错误处理的转录"""
    
    # 1. 检查文件存在
    if not os.path.exists(audio_path):
        print(f"错误：文件不存在 {audio_path}")
        return None
    
    # 2. 检查模型文件
    required_files = [
        "model/qwen3_asr_llm.q4_k.gguf",
        "model/qwen3_asr_encoder_frontend.int4.onnx",
        "model/qwen3_asr_encoder_backend.int4.onnx"
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误：模型文件缺失 {f}")
            return None
    
    config = ASREngineConfig(model_dir="model")
    engine = None
    
    try:
        engine = QwenASREngine(config)
        result = engine.transcribe(audio_path)
        return result
        
    except RuntimeError as e:
        if "辅助进程启动失败" in str(e):
            print("错误：辅助进程启动失败，检查 ONNX 模型")
        elif "模型加载失败" in str(e):
            print("错误：GGUF 模型加载失败，检查文件完整性")
        else:
            print(f"运行时错误：{e}")
        return None
        
    except FileNotFoundError as e:
        print(f"文件错误：{e}")
        return None
        
    except Exception as e:
        print(f"未知错误：{e}")
        return None
        
    finally:
        if engine:
            engine.shutdown()
```

### 8.2 调试模式

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

config = ASREngineConfig(
    model_dir="model",
    verbose=True  # 打印详细信息
)

engine = QwenASREngine(config)
```

### 8.3 性能分析

```python
import cProfile
import pstats

def profile_transcribe():
    config = ASREngineConfig(model_dir="model")
    engine = QwenASREngine(config)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = engine.transcribe("test.mp3")
    
    profiler.disable()
    
    # 输出性能统计
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 显示前 20 个耗时函数
    
    engine.shutdown()

profile_transcribe()
```

---

## 9. 性能优化

### 9.1 显存优化

```python
# 方案 1：减小上下文窗口
config = ASREngineConfig(n_ctx=1024)  # 默认 2048

# 方案 2：减少记忆片段
config = ASREngineConfig(memory_num=0)  # 禁用记忆

# 方案 3：使用更小模型
# 使用 0.6B 而非 1.7B

# 方案 4：降低精度
# 使用 INT4 Encoder + Q4_K Decoder
```

### 9.2 速度优化

```python
# 方案 1：启用 GPU 加速
config = ASREngineConfig(use_dml=True)  # Windows
# 或
os.environ["GGML_VULKAN"] = "1"  # 跨平台

# 方案 2：增大 chunk_size (减少片段数量)
config = ASREngineConfig(chunk_size=60.0)  # 默认 40s

# 方案 3：禁用对齐 (如果不需要时间戳)
config = ASREngineConfig(enable_aligner=False)
```

### 9.3 显存占用参考

| 配置 | ASR Encoder | ASR Decoder | Aligner | 总计 |
|------|-------------|-------------|---------|------|
| DML (INT4+Q4_K) | 473MB | - | - | ~0.5GB |
| Vulkan (INT4+Q4_K) | 420MB | 1.6GB | 0.9GB | ~2.9GB |
| CPU | - | - | - | 系统内存 |

---

## 10. 常见问题

### Q1: 输出乱码或「!!!!」

**原因**: Intel 集显 FP16 计算溢出

**解决**:
```python
import os
os.environ["GGML_VULKAN_DISABLE_F16"] = "1"
```

### Q2: 显存不足

**解决**:
```python
# 1. 减小上下文
config.n_ctx = 1024

# 2. 禁用记忆
config.memory_num = 0

# 3. 使用 CPU 模式
config.use_dml = False
```

### Q3: 速度过慢

**解决**:
```python
# 1. 启用 GPU
config.use_dml = True

# 2. 增大切片
config.chunk_size = 60.0

# 3. 禁用对齐
config.enable_aligner = False
```

### Q4: 模型文件找不到

**检查**:
```python
from pathlib import Path

model_files = [
    "model/qwen3_asr_llm.q4_k.gguf",
    "model/qwen3_asr_encoder_frontend.int4.onnx",
    "model/qwen3_asr_encoder_backend.int4.onnx"
]

for f in model_files:
    print(f"{f}: {Path(f).exists()}")
```

### Q5: 音频格式不支持

**支持格式**:
- MP3, WAV, M4A, FLAC, OGG, WMA

**转换**:
```python
from pydub import AudioSegment

audio = AudioSegment.from_file("input.aac")
audio.export("output.wav", format="wav")
```

---

## 附录 A: API 参考

### ASREngineConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_dir | str | - | 模型目录 |
| encoder_frontend_fn | str | qwen3_asr_encoder_frontend.int4.onnx | 前端模型 |
| encoder_backend_fn | str | qwen3_asr_encoder_backend.int4.onnx | 后端模型 |
| llm_fn | str | qwen3_asr_llm.q4_k.gguf | LLM 模型 |
| use_dml | bool | False | DirectML 加速 |
| n_ctx | int | 2048 | 上下文窗口 |
| chunk_size | float | 40.0 | 切片秒数 |
| memory_num | int | 1 | 记忆片段数 |
| enable_aligner | bool | False | 启用对齐 |
| align_config | AlignerConfig | None | 对齐配置 |
| verbose | bool | True | 详细日志 |

### QwenASREngine.transcribe()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| audio_file | str | - | 音频文件路径 |
| language | str | None | 强制语言 |
| context | str | None | 上下文提示 |
| start_second | float | 0.0 | 开始位置 |
| duration | float | None | 处理时长 |
| temperature | float | 0.4 | 采样温度 |

### TranscribeResult

| 字段 | 类型 | 说明 |
|------|------|------|
| text | str | 转录文本 |
| alignment | ForcedAlignResult | 对齐结果 (可选) |
| performance | dict | 性能统计 |

---

## 附录 B: 支持的语言

Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian

---

**文档结束**
