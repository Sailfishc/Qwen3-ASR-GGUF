import os
import time
import numpy as np
import onnxruntime as ort
import pynvml

# ============================================================================
# 配置与初始化
# ============================================================================
MODEL_PATH = r"D:\qwen3-asr\model\qwen3_asr_encoder.fp16.onnx"

try:
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def get_gpu_used():
    if not HAS_NVML: return 0
    info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
    return info.used / 1024 / 1024  # MB

def get_feat_lengths(t_mel: int) -> int:
    """计算 Qwen3 Frontend 输出长度"""
    t_leave = t_mel % 100
    feat_len = (t_leave - 1) // 2 + 1
    out_len = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (t_mel // 100) * 13
    return int(out_len)

# ============================================================================
# 核心测试函数
# ============================================================================
def run_memory_test(session, duration):
    sr = 16000
    t_mel = int(duration * sr // 160)
    t_out = get_feat_lengths(t_mel)
    
    # 构建输入
    # 注意：使用 float16 以匹配 .fp16.onnx 模型
    dummy_mel = np.zeros((1, 128, t_mel), dtype=np.float16)
    dummy_mask = np.zeros((1, 1, t_out, t_out), dtype=np.float16)
    
    input_feed = {
        "input_features": dummy_mel,
        "attention_mask": dummy_mask
    }

    # 记录执行前状态
    vram_before = get_gpu_used()
    
    start_time = time.perf_counter()
    session.run(None, input_feed)
    end_time = time.perf_counter()
    
    # 记录执行后状态
    vram_after = get_gpu_used()
    increment = vram_after - vram_before
    
    print(f"  [时长 {duration:2d}s] | T_out: {t_out:4d} | 耗时: {end_time-start_time:.2f}s | 显存增量: {increment:8.2f} MB")
    return increment

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    print("\n" + "="*60)
    print(" Qwen3-ASR 显存增长趋势验证脚本 (Black-box Trend Analysis)")
    print("="*60)

    # 1. 载入模型 (只载入一次，避免反复载入干扰)
    sess_opts = ort.SessionOptions()
    # 限制显存管理器的碎片整理压力，使增量更明显
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    
    vram_init = get_gpu_used()
    print(f"正在载入模型...")
    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=providers)
    vram_after_load = get_gpu_used()
    
    load_inc = vram_after_load - vram_init
    print(f"模型载入完成，固定开销: {load_inc:.2f} MB")
    print("-" * 60)

    # 2. 梯度时长测试
    durations = [10, 20, 40]
    results = []

    print("开始压力测试:")
    for d in durations:
        inc = run_memory_test(session, d)
        results.append(inc)

    print("-" * 60)
    
    # 3. 分析结果
    # 10s -> 20s (翻倍)
    ratio_1_2 = results[1] / results[0] if results[0] > 0 else 0
    # 20s -> 40s (翻倍)
    ratio_2_3 = results[2] / results[1] if results[1] > 0 else 0

    print("分析结论:")
    print(f"  10s -> 20s 增长倍数: {ratio_1_2:.2f}x (理论: 线性=2x, 平方=4x)")
    print(f"  20s -> 40s 增长倍数: {ratio_2_3:.2f}x (理论: 线性=2x, 平方=4x)")

    avg_ratio = (ratio_1_2 + ratio_2_3) / 2
    
    if avg_ratio >= 3.0:
        print("\n结论抢先看: [确认] 显存爆炸主要源于 O(T^2) 操作。")
        print("这证明了 Attention Mask 和 Attention Weights 矩阵是罪魁祸首。")
    elif avg_ratio >= 1.5:
        print("\n结论抢先看: [确认] 显存增长是线性的 O(T)。")
        print("这说明问题不在 Attention 矩阵，而在每一层的特征图 (Feature Maps/Buffers) 堆积。")
    else:
        print("\n结论: 增长不明显，可能显存管理机制有缓存，请尝试重启后运行。")

    print("="*60 + "\n")

if __name__ == "__main__":
    main()
