import os
import time
import numpy as np
import onnxruntime as ort
import pynvml
from export_config import EXPORT_DIR

# ============================================================================
# 配置与初始化
# ============================================================================
MODEL_PATH = os.path.join(EXPORT_DIR, "qwen3_asr_frontend.onnx")

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

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到前端模型 {MODEL_PATH}")
        return

    print("\n" + "="*50)
    print(" 脚本 35: 前端 (Frontend) FP32 显存压力测试")
    print("="*50)

    # 1. 载入模型
    vram_start = get_gpu_used()
    sess_opts = ort.SessionOptions()
    # 按照 DML 优化逻辑配置
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    
    print(f"正在载入前端模型...")
    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    vram_after_load = get_gpu_used()
    print(f"载入后显存: {vram_after_load:.2f} MB (净开销: {vram_after_load - vram_start:.2f} MB)")

    # 2. 准备 40s 音频输入 (4000 帧 Mel)
    duration = 40
    t_mel = duration * 16000 // 160 
    dummy_mel = np.zeros((1, 128, t_mel), dtype=np.float32)
    
    print(f"\n正在模拟 {duration}s 音频前端推理 (T={t_mel})...")
    
    # 获取输入输出名
    input_name = session.get_inputs()[0].name
    
    # 热身运行 (让 DML 分配初始 Buffer)
    session.run(None, {input_name: dummy_mel})
    vram_post_warmup = get_gpu_used()
    
    # 正式推理并统计
    t_start = time.perf_counter()
    session.run(None, {input_name: dummy_mel})
    t_end = time.perf_counter()
    
    vram_final = get_gpu_used()
    increment = vram_final - vram_after_load
    
    print(f"推理耗时: {t_end - t_start:.4f}s")
    print(f"推理后总显存: {vram_final:.2f} MB")
    print(f"本次前端推理阶段显存增量: {increment:.2f} MB")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
