# coding=utf-8
import os
import numpy as np
import onnxruntime as ort
from export_config import EXPORT_DIR

def calculate_cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

def main():
    onnx_path = os.path.join(EXPORT_DIR, "qwen3_asr_encoder.onnx")
    input_mel_path = "capture_data/input_mel.npy"
    # 直接对比最终的编码器输出
    baseline_output_path = "capture_data/encoder_backend_output.npy"

    if not os.path.exists(onnx_path):
        print("❌ 缺失 ONNX 模型。")
        return
    if not os.path.exists(input_mel_path):
        print("❌ 缺失输入数据。")
        return

    # 1. 加载数据
    input_mel = np.load(input_mel_path) # (1, 128, 2850)
    baseline_output = np.load(baseline_output_path) # (371, 1024)
    
    seq_len_out = baseline_output.shape[0]
    
    # 2. 准备输入
    # 使用全屏 Mask
    mask_input = np.zeros((1, 1, seq_len_out, seq_len_out), dtype=np.float32)
    
    # 3. 推理 (DirectML)
    print(f"Initializing ONNX Runtime with DirectML...")
    sess = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    
    print(f"Running Full Encoder ONNX verification...")
    ort_outs = sess.run(None, {
        "input_features": input_mel,
        "attention_mask": mask_input
    })
    onnx_output = ort_outs[0][0] # (371, 1024)

    # 4. 对比
    sim = calculate_cosine_similarity(baseline_output, onnx_output)
    
    print("\n" + "="*40)
    print(f"Encoder 最终阅兵验证 (Combined Encoder Verification):")
    print(f"  - ONNX 输出形状: {onnx_output.shape}")
    print(f"  - 官方基准形状: {baseline_output.shape}")
    print(f"  - 最终余弦相似度: {sim:.8f}")
    
    if sim > 0.999999:
        print("\n🏆 PERFECT SCORE! The combined ASR Encoder is a mirror image of the official model.")
    elif sim > 0.999:
        print("\n✅ SUCCESS: High precision verified.")
    else:
        print("\n❌ FAILED: Significant discrepancy detected.")
    print("="*40)

if __name__ == "__main__":
    main()
