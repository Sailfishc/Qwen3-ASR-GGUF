# coding=utf-8
import os
import time
import numpy as np
import onnxruntime as ort
import librosa

class FastWhisperMel:
    """基于 NumPy 和 Librosa 的 Mel 提取器"""
    def __init__(self, filter_path: str):
        self.filters = np.load(filter_path) # (201, 128)
        
    def __call__(self, audio: np.ndarray, dtype=np.float32) -> np.ndarray:
        # 1. STFT
        stft = librosa.stft(audio, n_fft=400, hop_length=160, window='hann', center=True)
        # 2. 能量谱
        magnitudes = np.abs(stft) ** 2
        # 3. Mel 映射
        mel_spec = np.dot(self.filters.T, magnitudes)
        # 4. 取对数
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        # 5. 归一化
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        # 6. 帧对齐：丢弃 stft(center=True) 产生的多余帧
        n_frames = audio.shape[-1] // 160
        log_spec = log_spec[:, :n_frames]
        return log_spec.astype(dtype)

def get_feat_lengths(t_mel: int) -> int:
    """计算下采样后的特征长度 (与官方 C++ 版一致)"""
    t_leave = t_mel % 100
    feat_len = (t_leave - 1) // 2 + 1
    out_len = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (t_mel // 100) * 13
    return int(out_len)

class QwenAudioEncoder:
    """Qwen3 音频编码器 (ONNX 后端)"""
    def __init__(self, encoder_path: str, mel_filters_path: str, use_dml: bool = True, warmup_sec: float = 5.0, verbose: bool = True):
        self.verbose = verbose
        
        # 初始化 ONNX Session
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if use_dml and 'DmlExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'DmlExecutionProvider') 
            
        if self.verbose: print(f"--- [Encoder] 加载 ONNX 模型 (DML: {use_dml}) ---")
        self.session = ort.InferenceSession(encoder_path, sess_options=sess_opts, providers=providers)
        self.mel_extractor = FastWhisperMel(mel_filters_path)
        
        # 检测精度
        try:
            fe_input_type = self.session.get_inputs()[0].type
            self.input_dtype = np.float16 if 'float16' in fe_input_type else np.float32
        except:
            self.input_dtype = np.float32

        # 预热选项
        if warmup_sec > 0:
            if self.verbose: print(f"--- [Encoder] 正在预热 ({warmup_sec}s 随机音频)... ---")
            dummy_wav = np.random.randn(int(16000 * warmup_sec)).astype(np.float32)
            _ = self.encode(dummy_wav)
            if self.verbose: print("--- [Encoder] 预热完成 ---")

    def encode(self, audio: np.ndarray) -> tuple:
        """执行编码，返回 (embedding, 耗时)"""
        t0 = time.time()
        
        # 1. 提取 Mel
        mel = self.mel_extractor(audio, dtype=self.input_dtype) 
        mel_input = mel[np.newaxis, ...]
        
        # 2. 构造 Mask
        t_mel = mel.shape[1]
        t_out = get_feat_lengths(t_mel)
        mask_input = np.zeros((1, 1, t_out, t_out), dtype=self.input_dtype)
        
        # 3. 执行推理
        audio_embd = self.session.run(None, {
            "input_features": mel_input,
            "attention_mask": mask_input
        })[0]
        
        if audio_embd.ndim == 3: 
            audio_embd = audio_embd[0]
            
        elapsed = time.time() - t0
        return audio_embd, elapsed
