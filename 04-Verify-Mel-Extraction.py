# coding=utf-8
import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from export_config import EXPORT_DIR

# ==========================================
# 2. 音频前端：FastWhisperMel (用户提供逻辑)
# ==========================================
class FastWhisperMel:
    def __init__(self, filter_path: str):
        self.filters = np.load(filter_path)
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """输入: PCM 采样 (1D float32), 输出: Mel 频谱 (128, T)"""
        # 1. 核心 STFT
        # 注意：Whisper 标准使用 n_fft=400, hop_length=160, window='hann', center=True
        stft = librosa.stft(audio, n_fft=400, hop_length=160, window='hann', center=True)
        magnitudes = np.abs(stft) ** 2
        
        # 2. 映射到 Mel 域
        # self.filters shape: (201, 128) -> filter.T @ magnitudes
        mel_spec = np.dot(self.filters.T, magnitudes)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        
        # 3. 标准化 (Whisper style: max-8.0, then scale)
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        
        # 4. 帧对齐：官方逻辑是丢弃 stft(center=True) 产生的多余帧
        # librosa center=True 会在两端各补 n_fft//2 个采样
        n_frames = audio.shape[-1] // 160
        log_spec = log_spec[:, :n_frames]
        
        return log_spec.astype(np.float32)

def calculate_cosine_similarity(a, b):
    """计算余弦相似度"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

def main():
    # 1. 路径准备
    filter_path = os.path.join(EXPORT_DIR, "mel_filters.npy")
    official_mel_path = "capture_data/input_mel.npy"
    audio_path = "test.mp3"
    
    if not os.path.exists(filter_path):
        print(f"❌ 找不到滤波器文件: {filter_path}")
        return
    if not os.path.exists(official_mel_path):
        print(f"❌ 找不到官方捕获的 Mel 数据: {official_mel_path}")
        return

    # 2. 加载数据
    print(f"加载音频: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    
    print(f"加载官方数据: {official_mel_path}")
    official_mel = np.load(official_mel_path) # shape usually (B, 128, T)
    if official_mel.ndim == 3:
        official_mel = official_mel[0] # 取第一条 batch
    
    # 3. 自定义提取
    print("使用 FastWhisperMel 提取特征...")
    extractor = FastWhisperMel(filter_path)
    custom_mel = extractor(audio)
    
    # 4. 对齐长度 (由于预处理过程中可能存在微小的补齐差异)
    T_off = official_mel.shape[1]
    T_cus = custom_mel.shape[1]
    
    print(f"官方 Mel 形状: {official_mel.shape}")
    print(f"自定义 Mel 形状: {custom_mel.shape}")
    
    if T_off != T_cus:
        print(f"⚠️ 长度不一致: 官方 {T_off} vs 自定义 {T_cus}，正在尝试对齐...")
        min_T = min(T_off, T_cus)
        official_mel = official_mel[:, :min_T]
        custom_mel = custom_mel[:, :min_T]
    
    # 5. 计算相似度
    similarity = calculate_cosine_similarity(official_mel, custom_mel)
    
    print("\n" + "="*40)
    print(f"结果对比 (Cosine Similarity): {similarity:.6f}")
    if similarity > 0.999:
        print("✅ 验证通过：自定义实现与官方实现高度一致！")
    elif similarity > 0.95:
        print("⚠️ 验证基本通过：存在细微差别，可能由于浮点精度或窗函数实现细节引起。")
    else:
        print("❌ 验证未通过：相似度过低，请检查实现逻辑。")
    print("="*40)

    # 6. 保存自定义结果供后续参考
    np.save("capture_data/custom_mel.npy", custom_mel)
    print(f"自定义 Mel 已保存至 capture_data/custom_mel.npy")

if __name__ == "__main__":
    main()
