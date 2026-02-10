# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Qwen3ASRFrontendFullOnnx(nn.Module):
    """
    Qwen3-ASR 完整前端 (DirectML 深度优化版)
    根据 DML 经验文档 4.1 节优化：
    - 使用 unfold 替代 view 进行分块，避免动态维度退化。
    - 使用 flatten 替代 view(-1, ...)，通过维度继承绕开 Reshape 节点崩溃。
    - 维持符号化长度计算以支持任意长度输入。
    """
    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        self.register_buffer("pos_embed_table", audio_tower.positional_embedding.positional_embedding)
        
    def _get_feat_extract_output_lengths(self, input_lengths):
        """符号化长度计算策略"""
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def forward(self, input_features: torch.Tensor):
        # input_features: (Batch, 128, Time)
        # 1. 动态长度获取
        # 使用 input_features.shape[2] 确保符号化追踪
        t = input_features.shape[2] 
        chunk_size = 100
        
        # 2. 计算期望长度
        expected_len = self._get_feat_extract_output_lengths(t)
        
        # 3. 补齐到 100 的倍数
        pad_len = (chunk_size - (t % chunk_size)) % chunk_size
        x = F.pad(input_features, (0, pad_len)) # (Batch, 128, T_padded)
        
        # 4. 【核心优化】使用 unfold 进行分块处理 (代替 view)
        # (B, 128, T_padded) -> (B, 128, N_chunks, 100)
        x = x.unfold(2, chunk_size, chunk_size)
        
        # 维度重排: (B, 128, N, 100) -> (B, N, 128, 100)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # 展平前两维 (B, N) 到 Batch 维: (B*N, 1, 128, 100)
        # flatten 比 view 更有利于 DML 自动推断内存布局
        x = x.flatten(0, 1).unsqueeze(1)
        
        # 5. 三层卷积下采样
        x = F.gelu(self.conv2d1(x)) # (Batch_total, 480, 64, 50)
        x = F.gelu(self.conv2d2(x)) # (Batch_total, 480, 32, 25)
        x = F.gelu(self.conv2d3(x)) # (Batch_total, 480, 16, 13)
        
        # 6. 【核心优化】维度继承投影 (展平投影轴)
        # (N, 480, 16, 13) -> (N, 13, 480, 16)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 使用 flatten(2, 3) 代替显式的特征维 reshape
        # 480 * 16 = 7680，后端已知为常数
        x = x.flatten(2, 3) 
        x = self.conv_out(x) # (N, 13, 896)
        
        # 7. 添加位置编码 (常量索引切片，DML 安全)
        pos_embed = self.pos_embed_table[:13, :].unsqueeze(0)
        x = x + pos_embed
        
        # 8. 【核心优化】还原大序列 (1, -1, 896)
        # 将前两维 N*13 展平，然后加回 Batch 维
        x = x.flatten(0, 1).unsqueeze(0)
        
        # 9. 精确裁剪
        # 虽然这属于数据依赖型切片，但在 DML 节点中属于 Slice 而非 Reshape，
        # 只要前面的 Reshape 部分通过继承变稳定了，这里大概率能跑
        x = x[:, :expected_len, :]
        
        return x

class Qwen3ASRAudioAttentionOnnx(nn.Module):
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
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output

class Qwen3ASRBackendOnnx(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        self.layers = nn.ModuleList()
        for raw_layer in audio_tower.layers:
            # Monkey batch
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
