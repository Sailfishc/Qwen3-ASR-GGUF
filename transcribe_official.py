#!/usr/bin/env python3
# coding=utf-8
"""
使用 Qwen3-ASR-0.6B 模型转录音频文件
直接调用官方 qwen_asr 包，无需手动转换模型
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from qwen_asr import Qwen3ASRModel

def transcribe(audio_path: str, model_name: str = "Qwen/Qwen3-ASR-0.6B"):
    """
    使用官方 Qwen3-ASR 模型转录音频
    
    Args:
        audio_path: 音频文件路径
        model_name: 模型名称，默认使用 0.6B
    """
    print(f"🎤 开始转录：{audio_path}")
    print(f"📦 使用模型：{model_name}")
    
    # 加载模型（会自动从 ModelScope 下载）
    print("⏳ 正在加载模型（首次运行需要下载）...")
    asr = Qwen3ASRModel.from_pretrained(
        model_name,
        device_map="cpu",  # 使用 CPU
        dtype=torch.float32
    )
    
    print("🚀 开始转录...")
    
    # 执行转录
    results = asr.transcribe(
        audio=audio_path,
        language=None,  # 自动识别语言
        return_time_stamps=False
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("📝 转录文本:")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\n[片段 {i+1}]")
        print(f"语言：{result.language}")
        print(f"文本：{result.text}")
    
    print("="*60)
    
    # 合并所有文本
    full_text = " ".join([r.text for r in results])
    print(f"\n完整文本:\n{full_text}")
    
    return full_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 transcribe_official.py <音频文件> [模型名称]")
        print("示例：python3 transcribe_official.py test_audio.wav")
        print("      python3 transcribe_official.py test_audio.wav Qwen/Qwen3-ASR-0.6B")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-ASR-0.6B"
    
    import torch
    transcribe(audio_file, model_name)
