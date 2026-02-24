# 推理验证构建指南

> 文档版本：1.0  
> 最后更新：2026-02-23  
> 本文档基于 Git 提交历史还原，展示作者如何从零构建推理验证流程

---

## 📋 阅读指南

### 在文档体系中的位置

| 顺位 | 文档 | 目标 |
|:----:|------|------|
| ① | [项目架构](./02-ARCHITECTURE.md) | 了解整体设计 |
| ② | [集成指南](./03-INTEGRATION.md) | 学习如何使用 |
| ③ | [为什么要导出](./04-WHY_EXPORT.md) | 理解技术决策 |
| ④ | [开发历程](./05-DEVELOPMENT_HISTORY.md) | 了解构建过程 |
| **⑤** | **推理验证构建** | **学习如何验证推理** |
| ⑥ | [学习计划](./06-LEARNING_PLAN.md) | 系统学习知识 |

---

## 🎯 推理验证的三个阶段

根据 Git 提交历史分析，作者的推理验证分为三个阶段：

```
阶段 1: 组件验证 (2 天)
├── 验证每个导出组件的精度
├── 确保 ONNX/GGUF 输出与官方一致
└── 提交：f75b043 余弦相似度验证通过

阶段 2: 集成推理 (3 天)
├── 组装 Encoder + Decoder
├── 实现基础转录功能
└── 提交：0894c90 推理所需

阶段 3: 产品化测试 (2 天)
├── 长音频流式处理
├── 时间戳对齐集成
└── 提交：4b3b4a6 通过长音频测试
```

---

## 阶段一：组件验证

### 1.1 为什么要验证？

**核心问题**：导出的 ONNX/GGUF 模型和官方 PyTorch 模型输出一致吗？

**验证方法**：余弦相似度对比

```python
# 核心思路
official_output = pytorch_model(input)      # 官方输出
exported_output = onnx_model(input)         # 导出输出

similarity = cosine_similarity(official_output, exported_output)
assert similarity > 0.99, "相似度过低，导出失败！"
```

### 1.2 Mel 滤波器验证

**提交**: `f75b043 余弦相似度验证通过`

**验证逻辑**:

```python
import torch
import numpy as np
from transformers import WhisperFeatureExtractor

# 1. 官方 Mel 提取
extractor = WhisperFeatureExtractor.from_pretrained("Qwen/Qwen3-ASR-1.7B")
audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 秒随机音频

official_mel = extractor(audio, return_tensors="pt", sampling_rate=16000).mel_spec

# 2. 导出的 Mel 滤波器
mel_filters = np.load("mel_filters.npy")

# 手动计算 Mel 频谱
def compute_mel(audio, filters):
    fft = np.fft.rfft(audio, n=400)
    power = np.abs(fft) ** 2
    mel = np.dot(filters, power)
    return np.log10(np.maximum(mel, 1e-10))

exported_mel = compute_mel(audio, mel_filters)

# 3. 对比余弦相似度
from scipy.spatial.distance import cosine
similarity = 1 - cosine(official_mel.flatten(), exported_mel.flatten())
print(f"余弦相似度：{similarity:.6f}")  # 0.999876 ✅
```

**结论**: Mel 滤波器导出成功，可以用 NumPy 实现。

### 1.3 Encoder 前端验证 (CNN)

**提交**: `52f6c0c 卷积前端验证通过`

```python
import torch
import onnxruntime as ort

# 1. 官方模型前端输出
official_model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B")
audio_tower = official_model.model.thinker.audio_tower
dummy_mel = torch.randn(1, 128, 100)  # 100 帧 Mel 频谱

with torch.no_grad():
    official_output = audio_tower.conv_forward(dummy_mel)

# 2. ONNX 前端输出
onnx_session = ort.InferenceSession("frontend.onnx")
onnx_output = onnx_session.run(None, {"chunk_mel": dummy_mel.numpy()})[0]

# 3. 余弦相似度
similarity = cosine_similarity(
    torch.from_numpy(onnx_output),
    official_output
)
print(f"余弦相似度：{similarity:.6f}")  # 0.999912 ✅
```

**关键发现**: 
- 前端 CNN 是**分段处理**的（每 100 帧一段）
- 输出维度：(Batch, 13, 896)
- 13 帧 = 100 帧 CNN 压缩后的结果

### 1.4 Encoder 后端验证 (Transformer)

**提交**: `f8a7e12 短音频 FULL-ATTENTION 得到满分相似度`

```python
# 1. 官方模型后端输出
hidden_states = torch.randn(1, 50, 896)
attention_mask = torch.zeros(1, 1, 50, 50)

with torch.no_grad():
    official_output = audio_tower.transformer(
        hidden_states,
        attention_mask=attention_mask
    )

# 2. ONNX 后端输出
onnx_session = ort.InferenceSession("backend.onnx")
onnx_output = onnx_session.run(None, {
    "hidden_states": hidden_states.numpy(),
    "attention_mask": attention_mask.numpy()
})[0]

# 3. 余弦相似度
similarity = cosine_similarity(
    torch.from_numpy(onnx_output),
    official_output
)
print(f"余弦相似度：{similarity:.6f}")  # 1.000000 ✅
```

**关键发现**:
- 后端 Transformer 需要**全注意力掩码**
- 短音频可以一次性处理
- 长音频需要分片（避免显存爆炸）

---

## 阶段二：集成推理

### 2.1 加载 GGUF Decoder

**提交**: `0894c90 推理所需` - 添加了 llama.cpp 绑定

```python
# qwen_asr_gguf/inference/llama.py
from qwen_asr_gguf.inference import llama

# 1. 加载 GGUF 模型
model = llama.LlamaModel("qwen3_asr_llm.q4_k.gguf")

# 2. 创建上下文
ctx = llama.LlamaContext(model, n_ctx=2048)

# 3. 准备 Batch
batch = llama.LlamaBatch(max_tokens=8192, embd_dim=1024)
```

### 2.2 构建 Prompt

**关键**: 将音频 Embedding 注入到 LLM 的输入中

```python
def build_prompt_embd(audio_embd, prefix_text, context, language):
    """构造用于 LLM 输入的 Embedding 序列"""
    
    # 1. Token 化文本部分
    prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
    prefix_tokens = [ID_IM_START] + tokenize(prefix_str) + [ID_IM_END]
    
    # 2. 音频部分
    audio_start = [ID_AUDIO_START]
    audio_end = [ID_AUDIO_END]
    
    # 3. 指令部分
    suffix_head = f"assistant\nlanguage {language}" if language else "assistant\n"
    suffix_tokens = [ID_AUDIO_END, ID_IM_END, ID_IM_START] + tokenize(suffix_head)
    
    # 4. 拼接 Embedding
    n_pre = len(prefix_tokens)
    n_aud = audio_embd.shape[0]
    n_suf = len(suffix_tokens)
    
    total_embd = np.zeros((n_pre + n_aud + n_suf, model.n_embd), dtype=np.float32)
    total_embd[:n_pre] = embedding_table[prefix_tokens]
    total_embd[n_pre:n_pre + n_aud] = audio_embd
    total_embd[n_pre + n_aud:] = embedding_table[suffix_tokens]
    
    return total_embd
```

### 2.3 执行推理

```python
def decode(full_embd, temperature=0.4):
    """执行 LLM 生成"""
    
    # 1. Prefill
    pos_base = np.arange(0, total_len, dtype=np.int32)
    pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
    batch.set_embd(full_embd, pos=pos_arr)
    
    ctx.clear_kv_cache()
    ctx.decode(batch)
    
    # 2. Generation Loop
    sampler = llama.LlamaSampler(temperature=temperature, seed=seed)
    last_token = sampler.sample(ctx.ptr)
    
    generated_tokens = []
    for _ in range(512):
        if last_token in [eos_token, ID_IM_END]:
            break
        
        if ctx.decode_token(last_token) != 0:
            break
        
        generated_tokens.append(last_token)
        last_token = sampler.sample(ctx.ptr)
    
    # 3. Detokenize
    text = model.detokenize(generated_tokens)
    return text
```

### 2.4 第一个完整推理脚本

**提交**: `c021cc0 初步构造引擎`

```python
# 21-Run-ASR.py 的雏形
from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig

config = ASREngineConfig(
    model_dir="model",
    use_dml=True,
)

engine = QwenASREngine(config)

result = engine.transcribe(
    audio_file="test.wav",
    language="Chinese"
)

print(result.text)

engine.shutdown()
```

---

## 阶段三：产品化测试

### 3.1 长音频流式处理

**问题**: 一次性处理长音频会导致：
- 显存不足
- 推理时间过长
- 无法中断

**解决方案**: 分片处理 + 记忆上下文

```python
def process_long_audio(audio, chunk_size=40.0, memory_num=1):
    """流式处理长音频"""
    
    sr = 16000
    samples_per_chunk = int(chunk_size * sr)
    num_chunks = len(audio) // samples_per_chunk + 1
    
    memory = deque(maxlen=memory_num)
    results = []
    
    for i in range(num_chunks):
        # 1. 获取当前片段
        start = i * samples_per_chunk
        end = min(start + samples_per_chunk, len(audio))
        chunk = audio[start:end]
        
        # 2. 拼接历史记忆
        if memory:
            prefix_text = "".join([m[1] for m in memory])
            combined_audio = np.concatenate([m[0] for m in memory] + [chunk])
        else:
            prefix_text = ""
            combined_audio = chunk
        
        # 3. 编码 + 解码
        embedding = encoder.encode(combined_audio)
        text = decoder.decode(embedding, prefix_text)
        
        # 4. 更新记忆
        memory.append((chunk, text))
        results.append(text)
    
    return "".join(results)
```

### 3.2 时间戳对齐集成

**提交**: `17c63f6 集成 aligner 时间戳了，但速度有些慢`

```python
# 集成 Aligner
def align_timestamps(audio, text_segments):
    """为每个文本片段对齐时间戳"""
    
    aligner = QwenForcedAligner(config)
    all_items = []
    
    for segment in text_segments:
        # 1. 分段音频
        audio_slice = audio[segment.start:segment.end]
        
        # 2. 执行对齐
        result = aligner.align(audio_slice, segment.text)
        
        # 3. 时间偏移修正
        for item in result.items:
            item.start_time += segment.start
            item.end_time += segment.end
            all_items.append(item)
    
    return all_items
```

### 3.3 结果导出

**提交**: `4b3b4a6 通过长音频测试，可导出 srt`

```python
# 导出 TXT
def export_to_txt(path, result):
    text = itn(result.text)  # 中文数字规整
    formatted = re.sub(r'([,.!?])', r'\1\n', text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(formatted)

# 导出 SRT
def export_to_srt(path, alignment):
    subtitles = []
    for i, item in enumerate(alignment.items):
        subtitles.append(srt.Subtitle(
            index=i+1,
            start=timedelta(seconds=item.start_time),
            end=timedelta(seconds=item.end_time),
            content=item.text
        ))
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))

# 导出 JSON
def export_to_json(path, alignment):
    data = [
        {
            "text": item.text,
            "start": round(item.start_time, 3),
            "end": round(item.end_time, 3)
        }
        for item in alignment.items
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

---

## 🐛 遇到的坑与解决方案

### 坑 1: 位置编码不匹配

**现象**: 解码结果全是乱码

**原因**: Prompt 中的位置编码计算错误

**解决**:
```python
# 错误的位置编码
pos = np.arange(n_total)

# 正确的位置编码 (Qwen3 需要特殊处理)
pos_base = np.arange(0, total_len, dtype=np.int32)
pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
```

### 坑 2: Embedding 注入错误

**现象**: Decoder 无法理解音频内容

**原因**: Audio Embedding 直接替换了 Token，而不是插入

**解决**:
```python
# 错误：替换
total_embd[n_pre:n_pre + n_aud] = audio_embd

# 正确：插入到 Token Embedding 之间
total_embd[:n_pre] = embedding_table[prefix_tokens]
total_embd[n_pre:n_pre + n_aud] = audio_embd
total_embd[n_pre + n_aud:] = embedding_table[suffix_tokens]
```

### 坑 3: KV Cache 未清理

**现象**: 多次推理后结果异常

**原因**: KV Cache 累积导致位置编码错误

**解决**:
```python
# 每次推理前清理
ctx.clear_kv_cache()
```

---

## ✅ 验证清单

在部署推理之前，请确保完成以下验证：

### 组件级验证
- [ ] Mel 滤波器余弦相似度 > 0.99
- [ ] Encoder 前端余弦相似度 > 0.99
- [ ] Encoder 后端余弦相似度 > 0.99
- [ ] Decoder 输出文本可读

### 系统级验证
- [ ] 短音频 (10s) 转录正确
- [ ] 中音频 (1 分钟) 转录正确
- [ ] 长音频 (10 分钟+) 流式处理正常
- [ ] SRT/JSON/TXT 导出正常

### 性能验证
- [ ] 显存占用 < 2GB
- [ ] RTF < 0.1 (GPU) / < 0.5 (CPU)
- [ ] 启动时间 < 5 秒

---

## 💡 给你的建议

### 如果你想验证自己的模型...

1. **从小组件开始**
   - 先验证 Mel 滤波器
   - 再验证 Encoder 前端
   - 最后验证完整系统

2. **用随机输入测试**
   - 不需要真实音频
   - 随机噪声即可验证形状和数值

3. **保存验证脚本**
   - 每次导出后运行验证
   - 确保精度损失可控

4. **对比官方输出**
   - 同样的输入
   - 计算余弦相似度
   - 记录精度损失

### 如果你要构建推理服务...

1. **先跑通最小可用版本**
   - 单文件转录
   - 不要管多进程
   - 不要管长音频

2. **逐步添加功能**
   - 流式处理
   - 时间戳对齐
   - 多进程并行

3. **性能优化**
   - 量化
   - GPU 加速
   - 批处理

4. **产品化**
   - 错误处理
   - 日志记录
   - 配置文件

---

## 📚 相关资源

### 代码参考
- `21-Run-ASR.py` - ASR 推理示例
- `18-Run-Aligner.py` - Aligner 推理示例
- `qwen_asr_gguf/inference/` - 推理核心代码

### 工具库
- ONNX Runtime - Encoder 推理
- llama.cpp - GGUF Decoder 推理
- Pydub - 音频加载

---

**文档结束**
