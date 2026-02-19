import os
import time
import numpy as np
import onnxruntime
import subprocess
import psutil

import pynvml

# 初始化 pynvml
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def get_vram_usage():
    """
    使用 pynvml 获取当前进程在 NVIDIA GPU 上的显存占用。
    如果不可用，退回到系统内存。
    """
    if not HAS_NVML:
        process = psutil.Process(os.getpid())
        return f"System RAM (RSS): {process.memory_info().rss / 1024 / 1024:.2f} MB (NVML not initialized)"

    try:
        pid = os.getpid()
        device_count = pynvml.nvmlDeviceGetCount()
        results = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            
            # 获取所有进程
            try:
                compute_p = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except:
                compute_p = []
            try:
                graphics_p = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except:
                graphics_p = []
            
            all_p = compute_p + graphics_p
            p_found = False
            for p in all_p:
                if p.pid == pid:
                    vram = p.usedGpuMemory
                    if vram is not None:
                        results.append(f"GPU {i} ({name}) Process VRAM: {vram / 1024 / 1024:.2f} MB")
                        p_found = True
                        break
            
            if not p_found:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                results.append(f"GPU {i} ({name}) Total Used: {info.used / 1024 / 1024:.2f} MB (Proc Memory Info Unavailable via NVML)")
        
        return "\n".join(results)
    except Exception as e:
        import traceback
        process = psutil.Process(os.getpid())
        return f"System RAM (RSS): {process.memory_info().rss / 1024 / 1024:.2f} MB\n(NVML Info: {e})"

def main():
    model_path = r"c:\Users\Haujet\Desktop\funasr-gguf\model\Fun-ASR-Nano-Encoder-Adaptor.fp16.onnx"
    
    print("\n--- [步骤 0] 载入模型前 ---")
    print(get_vram_usage())

    # 配置 ONNX Runtime 选项
    session_opts = onnxruntime.SessionOptions()
    # 按照 04-Inference.py 中的 nano_onnx.py 逻辑配置
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    
    print(f"\n--- [步骤 1] 正在载入模型 (DML) ---")
    t_start = time.perf_counter()
    try:
        session = onnxruntime.InferenceSession(
            model_path, 
            sess_options=session_opts, 
            providers=providers
        )
        print(f"模型载入成功，耗时: {time.perf_counter() - t_start:.2f}s")
        print(f"当前使用的 Provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"模型载入失败: {e}")
        return

    print("\n--- [步骤 2] 载入模型后 (进行任何计算前) ---")
    print(get_vram_usage())

    # 准备 40 秒音频
    SR = 16000
    duration = 40
    num_samples = SR * duration
    print(f"\n--- [步骤 3] 正在运行 {duration}s 音频推理 ---")
    
    # 构造输入
    # 根据 nano_onnx.py，输入可能是 float16 或 float32
    audio_type = session.get_inputs()[0].type
    dtype = np.float16 if 'float16' in audio_type else np.float32
    dummy_audio = np.zeros((1, 1, num_samples), dtype=dtype)
    dummy_ilens = np.array([num_samples], dtype=np.int64)
    
    in_names = [x.name for x in session.get_inputs()]
    out_names = [x.name for x in session.get_outputs()]
    
    input_feed = {}
    if len(in_names) >= 1:
        input_feed[in_names[0]] = dummy_audio
    if 'ilens' in in_names:
        input_feed['ilens'] = dummy_ilens
    elif len(in_names) >= 2:
        input_feed[in_names[1]] = dummy_ilens

    t_run = time.perf_counter()
    session.run(out_names, input_feed)
    print(f"推理完成，耗时: {time.perf_counter() - t_run:.2f}s")

    print("\n--- [步骤 4] 运行推理后 ---")
    print(get_vram_usage())

if __name__ == "__main__":
    main()
