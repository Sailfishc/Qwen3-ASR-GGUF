# 快速转录指南

> 使用 Qwen3-ASR-0.6B 模型转录音频文件

---

## 📋 文件信息

**待转录文件**: 
```
/Users/zhangcheng/Downloads/recording-2026-02-23T14-11-34-581Z-d46e76c4.webm
文件大小：251KB
格式：WebM (需要转换为 WAV)
```

---

## 步骤 1: 安装依赖

```bash
# 进入项目目录
cd /Users/zhangcheng/CodeProjects/Qwen3-ASR-GGUF

# 安装基础依赖
pip3 install torch torchaudio --break-system-packages
pip3 install onnxruntime-silicon --break-system-packages  # macOS Silicon
# 或
pip3 install onnxruntime --break-system-packages  # 通用版本

# 安装其他依赖
pip3 install -r requirements.txt --break-system-packages
```

---

## 步骤 2: 下载模型

### 方式 A: 下载预转换模型（推荐）

从 GitHub Releases 下载已转换好的 0.6B 模型：

```bash
# 创建模型目录
mkdir -p model

# 下载 0.6B 模型（需要手动下载）
# 访问：https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models
# 下载 "qwen3-asr-0.6b-gguf.zip" 或类似文件
# 解压到 model/ 目录
```

**所需模型文件**：
```
model/
├── qwen3_asr_llm.q4_k.gguf           # Decoder (约 400MB)
├── qwen3_asr_encoder_frontend.int4.onnx  # Encoder 前端 (约 10MB)
├── qwen3_asr_encoder_backend.int4.onnx   # Encoder 后端 (约 50MB)
└── mel_filters.npy                       # Mel 滤波器 (可选)
```

### 方式 B: 从官方模型转换

```bash
# 1. 安装 modelscope
pip3 install modelscope --break-system-packages

# 2. 下载官方 0.6B 模型
modelscope download --model Qwen/Qwen3-ASR-0.6B

# 3. 配置 export_config.py
# 编辑 ASR_MODEL_DIR 为下载路径

# 4. 执行转换（耗时约 30 分钟）
python3 01-Export-ASR-Encoder-Frontend.py
python3 02-Export_ASR-Encoder-Backend.py
python3 03-Optimize-ASR-Encoder.py
python3 04-Quantize-ASR-Encoder.py
python3 05-Export-ASR-Decoder-HF.py
python3 06-Convert-ASR-Decoder-GGUF.py
python3 07-Quantize-ASR-Decoder-GGUF.py
```

---

## 步骤 3: 转换音频格式

WebM 格式需要转换为 WAV：

```bash
# 使用 ffmpeg 转换
ffmpeg -i ~/Downloads/recording-2026-02-23T14-11-34-581Z-d46e76c4.webm \
       -ar 16000 -ac 1 \
       ./test_audio.wav
```

或者使用 Python 脚本：

```python
# convert_audio.py
from pydub import AudioSegment

# 加载 WebM 文件
audio = AudioSegment.from_file(
    "/Users/zhangcheng/Downloads/recording-2026-02-23T14-11-34-581Z-d46e76c4.webm"
)

# 转换为 16kHz 单声道 WAV
audio = audio.set_frame_rate(16000).set_channels(1)

# 保存
audio.export("test_audio.wav", format="wav")
print("✅ 音频转换完成：test_audio.wav")
```

---

## 步骤 4: 执行转录

### 方式 A: 使用命令行工具（推荐）

```bash
# 基本转录
python3 transcribe.py test_audio.wav -m ./model --prec int4 -y

# 带详细输出
python3 transcribe.py test_audio.wav \
    --model-dir ./model \
    --prec int4 \
    --language Chinese \
    --verbose \
    -y
```

### 方式 B: 使用 Python 脚本

创建 `quick_transcribe.py`：

```python
#!/usr/bin/env python3
# quick_transcribe.py

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, exporters

def transcribe(audio_path: str, model_dir: str = "model"):
    """快速转录函数"""
    
    print(f"🎤 开始转录：{audio_path}")
    print(f"📂 模型目录：{model_dir}")
    
    # 配置引擎 (0.6B 模型)
    config = ASREngineConfig(
        model_dir=model_dir,
        use_dml=False,  # macOS 不使用 DML
        enable_aligner=True,  # 启用时间戳对齐
        verbose=True
    )
    
    # 初始化引擎
    print("⚙️  正在加载模型...")
    engine = QwenASREngine(config)
    
    # 执行转录
    print("🎯 开始转录...")
    result = engine.transcribe(
        audio_file=audio_path,
        language="Chinese",  # 或 None 自动识别
        context=""  # 上下文提示
    )
    
    # 输出结果
    print("\n" + "="*50)
    print("📝 转录文本:")
    print("="*50)
    print(result.text)
    print("="*50)
    
    # 导出文件
    base_name = Path(audio_path).stem
    
    exporters.export_to_txt(f"{base_name}.txt", result)
    print(f"✅ 已保存文本：{base_name}.txt")
    
    if result.alignment:
        exporters.export_to_srt(f"{base_name}.srt", result)
        print(f"✅ 已保存字幕：{base_name}.srt")
        
        exporters.export_to_json(f"{base_name}.json", result)
        print(f"✅ 已保存时间戳：{base_name}.json")
    
    # 性能统计
    if result.performance:
        print("\n📊 性能统计:")
        print(f"  编码时间：{result.performance.get('encode_time', 0):.2f}s")
        print(f"  解码时间：{result.performance.get('decode_time', 0):.2f}s")
    
    # 清理
    engine.shutdown()
    return result.text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 quick_transcribe.py <音频文件>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "model"
    
    transcribe(audio_file, model_dir)
```

运行：
```bash
python3 quick_transcribe.py test_audio.wav model
```

---

## 预期输出

```
🎤 开始转录：test_audio.wav
📂 模型目录：model
⚙️  正在加载模型...
--- [QwenASR] 初始化引擎 (DML: False) ---
--- [QwenASR] 辅助进程已就绪 ---
--- [QwenASR] 引擎初始化耗时：2.50 秒 ---
🎯 开始转录...

[转录文本会在这里显示]

📊 性能统计:
  🔹 RTF (实时率) : 0.150 (越小越快)
  🔹 音频时长    : 15.20 秒
  🔹 总处理耗时  : 2.28 秒
  🔹 编码等待    : 0.15 秒
  🔹 LLM 预填充  : 0.320 秒 (856 tokens, 2675.0 tokens/s)
  🔹 LLM 生成    : 1.200 秒 (98 tokens, 81.7 tokens/s)
✅ 已保存文本文件：test_audio.txt
✅ 已生成字幕文件：test_audio.srt
✅ 已导出时间戳：test_audio.json
```

---

## 常见问题

### Q1: 找不到模型文件

**错误**: `错误：找不到以下所需模型文件`

**解决**: 
```bash
# 检查模型目录
ls -la model/

# 确认文件存在
model/
├── qwen3_asr_llm.q4_k.gguf           ✅
├── qwen3_asr_encoder_frontend.int4.onnx  ✅
└── qwen3_asr_encoder_backend.int4.onnx   ✅
```

### Q2: WebM 无法转换

**错误**: `ffmpeg: command not found`

**解决**:
```bash
# macOS 安装 ffmpeg
brew install ffmpeg

# 或使用 Python 转换
pip3 install pydub
python3 convert_audio.py
```

### Q3: 显存不足

**错误**: `CUDA out of memory`

**解决**:
```python
# 使用更小的上下文
config = ASREngineConfig(
    n_ctx=1024,  # 默认 2048
    memory_num=0,  # 禁用记忆
)
```

### Q4: 输出是乱码

**可能原因**:
1. 模型量化精度问题
2. 语言设置错误

**解决**:
```bash
# 尝试使用更高精度模型
python3 transcribe.py audio.wav --prec fp16

# 或强制指定语言
python3 transcribe.py audio.wav --language English
```

---

## 性能参考 (0.6B 模型)

| 设备 | RTF | 备注 |
|------|-----|------|
| M1/M2 Mac | 0.1-0.2 | CPU 推理 |
| NVIDIA GPU | 0.05-0.1 | CUDA 加速 |
| Intel CPU | 0.3-0.5 | 较慢 |

对于 251KB 的 WebM 文件（预计 15-30 秒音频）：
- 转换时间：~1 秒
- 转录时间：~3-6 秒 (M1/M2)
- 总耗时：~10 秒（含模型加载）

---

## 下一步

转录完成后，你可以：

1. **查看文本**: `cat test_audio.txt`
2. **查看字幕**: `cat test_audio.srt`
3. **查看时间戳**: `cat test_audio.json`
4. **编辑字幕**: 使用 Aegisub 等工具

---

**需要帮助？** 查看：
- [项目架构](./docs/ARCHITECTURE.md)
- [集成指南](./docs/INTEGRATION.md)
- [推理验证](./docs/INFERENCE_VALIDATION.md)
