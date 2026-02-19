import os
import time
import numpy as np
import onnxruntime as ort
import pynvml
import psutil

# 初始化 pynvml
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def get_gpu_info():
    """获取所有 GPU 的总已用显存"""
    if not HAS_NVML:
        return "NVML Not Available"
    
    results = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            results.append({
                "id": i,
                "name": name,
                "used": info.used / 1024 / 1024  # MB
            })
    except Exception as e:
        return f"Error: {e}"
    return results

def print_gpu_status(label):
    info = get_gpu_info()
    print(f"\n--- [{label}] ---")
    if isinstance(info, list):
        for g in info:
            print(f"GPU {g['id']} ({g['name']}): {g['used']:.2f} MB")
    else:
        print(info)

# 模拟 Qwen3 Encoder 的逻辑
def get_feat_lengths(t_mel: int) -> int:
    t_leave = t_mel % 100
    feat_len = (t_leave - 1) // 2 + 1
    out_len = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (t_mel // 100) * 13
    return int(out_len)

def main():
    model_path = r"D:\qwen3-asr\model\qwen3_asr_encoder.fp16.onnx"
    
    print_gpu_status("步骤 0: 载入模型前")

    # 1. 载入模型
    sess_opts = ort.SessionOptions()
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    
    print(f"\n正在载入 Qwen3 Encoder 模型 (DML)...")
    t_start = time.perf_counter()
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    print(f"载入完成 (耗时: {time.perf_counter() - t_start:.2f}s)")
    
    print_gpu_status("步骤 1: 载入模型后 (进行计算前)")

    # 2. 推理测试 (40秒音频)
    duration = 40
    sr = 16000
    print(f"\n正在模拟 {duration}s 音频推理...")
    
    # 模拟输入准备
    t_mel = int(duration * sr // 160)
    t_out = get_feat_lengths(t_mel)
    
    # 获取输入名称和类型
    inputs = session.get_inputs()
    input_dtype = np.float32
    for inp in inputs:
        if "float16" in inp.type:
            input_dtype = np.float16
            break
            
    dummy_mel = np.zeros((1, 128, t_mel), dtype=input_dtype)
    dummy_mask = np.zeros((1, 1, t_out, t_out), dtype=input_dtype)
    
    input_feed = {
        "input_features": dummy_mel,
        "attention_mask": dummy_mask
    }
    
    t_run = time.perf_counter()
    session.run(None, input_feed)
    print(f"推理完成 (耗时: {time.perf_counter() - t_run:.2f}s)")

    print_gpu_status("步骤 2: 推理完成后")

if __name__ == "__main__":
    main()
