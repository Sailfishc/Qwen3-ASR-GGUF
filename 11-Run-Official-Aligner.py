# coding=utf-8
import os
import torch
from qwen_asr import Qwen3ForcedAligner
from export_config import ALIGNER_MODEL_DIR

def main():
    # 1. 路径准备
    model_path = str(ALIGNER_MODEL_DIR)
    audio_path = "test.mp3"
    text_path = "test.txt"
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到 Aligner 模型目录: {model_path}")
        return
    if not os.path.exists(audio_path):
        print(f"❌ 找不到音频文件: {audio_path}")
        return
    if not os.path.exists(text_path):
        print(f"❌ 找不到文本文件: {text_path}")
        return

    # 2. 读取文本内容
    with open(text_path, "r", encoding="utf-8") as f:
        text_content = f.read().strip()

    print(f"--- 正在初始化 Aligner 模型 (Device: cuda) ---")
    # 注意：Aligner 官方推荐推理由其专门的 wrapper 处理
    aligner = Qwen3ForcedAligner.from_pretrained(
        model_path,
        dtype=torch.float32, # 强制 float32 以便后续对比精度
        device_map="cuda"
    )

    print(f"--- 正在执行强制对齐 (Forced Alignment) ---")
    # 执行对齐
    results = aligner.align(
        audio=audio_path,
        text=text_content,
        language="Chinese"
    )

    # 3. 输出结果
    print("\n" + "="*40)
    print("对齐结果预览:")
    print("="*40)
    
    # 结果是一个 ForcedAlignResult 列表，取第一个 sample
    sample_result = results[0]
    for i, item in enumerate(sample_result[:20]):
        print(f"[{i:02d}] {item.text: <4} | {item.start_time:6.3f}s -> {item.end_time:6.3f}s")
    
    print("...")
    print(f"总计对齐词数: {len(sample_result)}")
    print("="*40)

if __name__ == "__main__":
    main()
