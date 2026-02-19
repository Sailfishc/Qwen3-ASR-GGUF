# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Qwen3ASRFrontendFullOnnx(nn.Module):
    """
    Qwen3-ASR 完整前端 (DirectML 深度优化版)
    兼容 ASR (896) 和 Aligner (1024) 维度
    """
    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        # 记录输出维度，避免硬编码
        self.d_model = audio_tower.conv_out.out_features
        self.register_buffer("pos_embed_table", audio_tower.positional_embedding.positional_embedding)
        
    def _get_feat_extract_output_lengths(self, input_lengths):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def forward(self, input_features: torch.Tensor):
        # 1. 设置长度与填充 (维持 100 帧块对齐)
        t = input_features.shape[2] 
        chunk_size = 100
        expected_len = self._get_feat_extract_output_lengths(t)
        pad_len = (chunk_size - (t % chunk_size)) % chunk_size
        x = F.pad(input_features, (0, pad_len))
        
        # 2. 升维至 3D 模拟分块隔离 (Batch=1, Channel=1, Chunks=N, Freq=128, T_chunk=100)
        # 使用 unfold 物理隔离块，保证余弦相似度 1.0
        x = x.unfold(2, chunk_size, chunk_size) # (1, 128, num_chunks, 100)
        x = x.permute(0, 2, 1, 3).unsqueeze(1)  # (1, 1, num_chunks, 128, 100)
        
        # 3. 3D 卷积推理 (空间维度步长 2, 深度维度步长 1)
        # 由于卷积核在深度轴(Chunks)大小为 1，确保了块与块之间完全无数据泄漏
        x = F.gelu(F.conv3d(x, self.conv2d1.weight.unsqueeze(2), self.conv2d1.bias, stride=(1, 2, 2), padding=(0, 1, 1)))
        x = F.gelu(F.conv3d(x, self.conv2d2.weight.unsqueeze(2), self.conv2d2.bias, stride=(1, 2, 2), padding=(0, 1, 1)))
        x = F.gelu(F.conv3d(x, self.conv2d3.weight.unsqueeze(2), self.conv2d3.bias, stride=(1, 2, 2), padding=(0, 1, 1)))
        
        # 4. 维度变换与输出映射 (Batch=1, Chunks*T_out, Hidden)
        # 使用 permute + flatten 替代 view 以兼容 DML 动态形状
        x = x.permute(0, 2, 4, 1, 3).contiguous() # (1, Chunks, 13, C, F)
        x = x.flatten(1, 2) # (1, Chunks*13, C, F)
        x = x.flatten(2)    # (1, Chunks*13, D_conv)
        x = self.conv_out(x)
        
        # 5. 符号化位置编码 (依据经验文档 4.1 节，使用 cumsum 消除动态维度视图依赖)
        # 生成 [0..12, 0..12, ...] 循环索引
        t_out = x.shape[1]
        indices = (torch.ones(t_out, device=x.device, dtype=torch.long).cumsum(0) - 1) % 13
        pos_embed = self.pos_embed_table[indices, :].unsqueeze(0)
        x = x + pos_embed
        
        # 6. 对齐官方长度
        x = x[:, :expected_len, :]
        return x

class Qwen3ASRAudioAttentionOnnx(nn.Module):
    """
    Qwen3-ASR 多头注意力 (DML 友好 + 符号追踪修复版)
    """
    def __init__(self, raw_attn):
        super().__init__()
        self.num_heads = raw_attn.num_heads
        self.head_dim = raw_attn.head_dim
        self.scaling = raw_attn.scaling
        self.q_proj = raw_attn.q_proj
        self.k_proj = raw_attn.k_proj
        self.v_proj = raw_attn.v_proj
        self.out_proj = raw_attn.out_proj

    def forward(self, hidden_states, attention_mask=None):
        b, t, d = hidden_states.shape
        # 使用 unflatten/transpose 替代 view 
        # 经验文档建议：在 DML 中尽量保持 Batch=1 
        q = self.q_proj(hidden_states).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_proj(hidden_states).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        
        if attention_mask is not None:
            # 依据经验文档 4.2 节：使用 Additive Masking 替代昂贵的 masked_fill
            # DML 对 Add 算子的融合效果显著优于 Where (masked_fill)
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # 展平输出
        attn_output = attn_output.transpose(1, 2).contiguous().flatten(2)
        attn_output = self.out_proj(attn_output)
        return attn_output

class Qwen3ASRBackendOnnx(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.layers = nn.ModuleList()
        for raw_layer in audio_tower.layers:
            raw_layer.self_attn = Qwen3ASRAudioAttentionOnnx(raw_layer.self_attn)
            self.layers.append(raw_layer)
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.act = audio_tower.act
        self.proj2 = audio_tower.proj2
        
    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states = layer.self_attn(hidden_states, attention_mask=attention_mask)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.fc1(hidden_states)
            hidden_states = layer.activation_fn(hidden_states)
            hidden_states = layer.fc2(hidden_states)
            hidden_states = residual + hidden_states
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

class Qwen3ASREncoderFullOnnx(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.frontend = Qwen3ASRFrontendFullOnnx(audio_tower)
        self.backend = Qwen3ASRBackendOnnx(audio_tower)
        
    def forward(self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        hidden_states = self.frontend(input_features)
        last_hidden_state = self.backend(hidden_states, attention_mask=attention_mask)
        return last_hidden_state
