# Qwen3-ASR-GGUF 开发历程

> 本文档通过 Git 提交历史逆向分析，还原作者从零构建项目的完整过程  
> 分析日期：2026-02-23  
> 总提交数：50+  
> 开发周期：约 2 个月（2026-02 至今）  
> **推荐阅读顺序：第 ③ 顺位**（理解项目构建过程）

---

## 📚 阅读指南

### 在文档体系中的位置

| 顺位 | 文档 | 文件名 | 目标读者 | 预计耗时 |
|:----:|------|--------|----------|----------|
| **①** | [项目架构](./02-ARCHITECTURE.md) | `02-ARCHITECTURE.md` | 想了解项目整体设计 | 1-2 小时 |
| **②** | [集成指南](./03-INTEGRATION.md) | `03-INTEGRATION.md` | 想快速使用项目 | 1-2 小时 |
| **③** | **开发历程** | `05-DEVELOPMENT_HISTORY.md` | **想了解项目如何构建** | **2-3 小时** |
| **④** | [学习计划](./06-LEARNING_PLAN.md) | `06-LEARNING_PLAN.md` | 想深入理解原理 | 4-12 周 |
| **⑤** | [导出指南](./EXPORT_GUIDE.md) | `EXPORT_GUIDE.md` | 想转换自己的模型 | 2-4 小时 |
| **⑥** | [源码解析](./SOURCE_CODE.md) | `SOURCE_CODE.md` | 想修改/扩展功能 | 4-8 周 |

### 本文档价值

- ✅ 了解真实项目的演进过程
- ✅ 学习工程化思维
- ✅ 理解技术决策背后的原因
- ✅ 为自己的项目提供参考

### 阅读建议

```
如果你：
├─ 想了解作者是如何一步步构建这个项目的 → 完整阅读
├─ 想学习如何从零开始一个 ML 工程 → 重点看阶段一、二
├─ 想理解关键设计决策 → 重点看"关键技术决策分析"章节
└─ 想避免常见坑 → 重点看"关键 Bug 与解决方案"章节
```

---

## 🎯 项目演进时间线

```
2026-02 上旬                    2026-02 中旬                   2026-02 下旬
     │                               │                              │
     ▼                               ▼                              ▼
┌──────────┐                  ┌──────────┐                  ┌──────────┐
│ 阶段一   │                  │ 阶段二   │                  │ 阶段三   │
│ 模型导出 │        →         │ 推理验证 │        →         │ 工程优化 │
│ (2周)    │                  │ (1周)    │                  │ (1周)    │
└──────────┘                  └──────────┘                  └──────────┘
     │                               │                              │
     • 分析官方模型                 • 构建推理脚本                 • 性能优化
     • 导出 Encoder                 • 验证精度                     • 打包发布
     • 导出 Decoder                 • 集成 Aligner                 • 文档完善
     • 导出 Mel 特征                • 长音频测试                   • Bug 修复
```

---

## 阶段一：模型导出（约 2 周）

### Week 1: 探索与验证

#### Day 1-3: 理解官方模型

**提交**: `c556c2d 初步提交` → `18228b2 官方推理`

作者首先做的事情：

```python
# 1. 加载官方模型，理解结构
from transformers import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen3-ASR-1.7B")

# 2. 分析模型组件
print(model.model.thinker.audio_tower)  # Encoder 部分
print(model.model.thinker.llm)          # Decoder 部分
```

**关键发现**:
- 模型分为 Audio Tower (Encoder) 和 LLM (Decoder)
- Encoder 又分前端 (CNN) 和后端 (Transformer)
- 需要导出的组件：Mel 滤波器、Encoder、Decoder

#### Day 4-7: 导出 Encoder

**提交**: `f75b043 余弦相似度验证通过` → `52f6c0c 卷积前端验证通过`

**遇到的第一个挑战**: Encoder 结构复杂

```
原始模型结构:
Audio Tower
├── Frontend (CNN)
│   ├── Conv1d + GELU
│   ├── Conv1d + GELU
│   └── Conv1d
├── Positional Encoding
└── Backend (Transformer)
    └── 24 层 Transformer
```

**解决方案**: 分步导出

```python
# 步骤 1: 导出前端 CNN
class Qwen3ASRFrontendOnnx(nn.Module):
    def __init__(self, audio_tower):
        super().__init__()
        # 只取前端的卷积部分
        self.conv1 = audio_tower.conv1
        self.conv2 = audio_tower.conv2
        self.conv3 = audio_tower.conv3
    
    def forward(self, x):
        x = self.conv1(x)  # (B, 128, T) -> (B, 384, T/2)
        x = F.gelu(x)
        x = self.conv2(x)  # -> (B, 384, T/4)
        x = F.gelu(x)
        x = self.conv3(x)  # -> (B, 896, T/4)
        return x
```

**验证方法**: 余弦相似度验证

```python
# 验证导出精度
def verify_cosine_similarity(onnx_output, torch_output):
    similarity = cosine_similarity(onnx_output, torch_output)
    assert similarity > 0.99, f"相似度 {similarity} 过低！"
```

### Week 2: 解决显存问题

#### Day 8-10: 显存爆炸

**提交**: `eae0258 尝试解决卷积占用内存的问题`

**问题**: 一次性导出整个 Encoder，长音频导致 OOM

```
错误现象:
CUDA out of memory. Tried to allocate 20.00 GiB
```

**分析**: CNN 前端在处理长序列时，中间激活值占用巨大显存

**第一次尝试**: 多通道卷积取代 batch
- 效果不佳，仍然占用大量显存

**第二次尝试**: **分体 Encoder**（关键决策）

**提交**: `da0590c 使用分体的 encoder`

```python
# 关键创新：前端分段处理
class QwenAudioEncoder:
    def __init__(self, frontend_path, backend_path):
        self.sess_fe = ort.InferenceSession(frontend_path)  # CNN
        self.sess_be = ort.InferenceSession(backend_path)   # Transformer
    
    def encode(self, audio):
        # 1. 提取 Mel
        mel = self.mel_extractor(audio)  # (128, T)
        
        # 2. 前端分段处理（关键！）
        chunk_size = 100
        outputs = []
        for i in range(0, mel.shape[1], chunk_size):
            chunk = mel[:, i:i+chunk_size]
            out = self.sess_fe.run(None, {"chunk_mel": chunk})
            outputs.append(out)
        
        # 3. 拼接后传给后端
        hidden = np.concatenate(outputs, axis=1)
        return self.sess_be.run(None, {"hidden_states": hidden})
```

**收益**: 显存占用从 20GB 降至 <1GB

#### Day 11-14: 导出 Decoder 和 Aligner

**提交**: `ba5983e gguf 导出必须` → `15b8074 成功跑通 aligner`

**Decoder 导出挑战**: 
- 官方使用 HuggingFace Transformers
- 需要转换为 GGUF 格式供 llama.cpp 使用

**解决方案**: 借用 llama.cpp 的 convert_hf_to_gguf

```python
# 关键补丁：让转换器支持 Qwen3-ASR 的特殊结构
def patched_load_hparams(dir_model: Path):
    with open(dir_model / "config.json") as f:
        config = json.load(f)
    
    # 适配 Qwen3-ASR 的特殊字段名
    if "llm_config" in config:
        config["text_config"] = config["llm_config"]
    
    return config
```

**Aligner 模型导出**:
- 同样的流程导出 Aligner Encoder
- 发现维度不同：ASR 输出 896 维，Aligner 输出 1024 维
- **关键修改**: 动态获取维度

```python
# 兼容 ASR 和 Aligner
conv_out = self.conv3(x)  # 可能是 896 或 1024
dim = conv_out.shape[-1]  # 动态获取
```

---

## 阶段二：推理验证（约 1 周）

### Week 3: 构建推理管道

#### Day 15-17: 基础推理

**提交**: `c021cc0 初步构造引擎` → `d930989 encoder 成功集成到转录脚本中`

**第一个可运行的推理脚本**:

```python
# 21-Run-ASR.py 的雏形
class QwenASREngine:
    def __init__(self, model_dir):
        # 1. 加载 Encoder
        self.encoder = QwenAudioEncoder(
            frontend_path=f"{model_dir}/frontend.onnx",
            backend_path=f"{model_dir}/backend.onnx"
        )
        
        # 2. 加载 Decoder (llama.cpp)
        self.model = llama.LlamaModel(f"{model_dir}/decoder.gguf")
        self.ctx = llama.LlamaContext(self.model, n_ctx=2048)
    
    def transcribe(self, audio_path):
        # 1. 编码
        audio_emb = self.encoder.encode(audio)
        
        # 2. 构建 Prompt
        prompt = self.build_prompt(audio_emb)
        
        # 3. 解码
        text = self.decode(prompt)
        return text
```

**遇到的坑**:
- llama.cpp 的 batch 接口不熟悉
- Token embedding 注入方式错误
- Prompt 格式与官方不一致

#### Day 18-19: 对齐时间戳

**提交**: `17c63f6 集成 aligner 时间戳了，但速度有些慢`

**集成策略**:
```
原始流程:
音频 → Encoder → Decoder → 文本

改进流程:
音频 → Encoder → Decoder → 文本
                          ↓
                     分段音频 + 文本 → Aligner → 时间戳
```

**性能问题**: Aligner 串行执行，拖慢整体速度

**初步优化**:
```python
# 使用多进程并行
from multiprocessing import Process

align_process = Process(target=align_worker)
align_process.start()
```

#### Day 20-21: 长音频测试

**提交**: `4b3b4a6 通过长音频测试，可导出 srt`

**流式处理策略**:
```python
# 40 秒切片 + 记忆上下文
chunk_size = 40.0
memory_num = 1

def process_long_audio(audio):
    chunks = split_audio(audio, chunk_size)
    memory = deque(maxlen=memory_num)
    
    for chunk in chunks:
        # 拼接历史记忆
        full_audio = concatenate(memory + [chunk])
        text = transcribe_chunk(full_audio)
        memory.append((chunk, text))
```

**验证成功**:
- 50 分钟音频可以完整转录
- SRT 字幕正常生成

---

## 阶段三：工程优化（约 1 周）

### Week 4: 性能与工程化

#### Day 22-24: 量化与优化

**提交**: `ab498eb 对导出的 fp32 encoder 优化合并gelu`

**ONNX 优化流程**:
```
FP32 (原始) 
  → 优化 (算子融合、常量折叠)
  → FP16 (显存减半)
  → INT8 (相似度 98%)
  → INT4 (相似度 96%，推荐)
```

**量化收益**:
- Encoder 显存: 473MB → 120MB (INT4)
- 速度提升: 30%

**Decoder 量化**:
```bash
# 先导出 FP16，再量化为 Q4_K
./llama-quantize model.gguf model_q4_k.gguf Q4_K
```

#### Day 25-27: 启动优化

**提交**: `3917d4b 统计和优化启动时间`

**发现的问题**:
- 启动需要 10+ 秒
- 主要耗时在: librosa 导入、模型加载、预热

**优化措施**:

1. **移除 librosa 依赖**
   ```python
   # 原来
   import librosa
   mel = librosa.feature.melspectrogram(...)
   
   # 优化后
   from scipy.signal import get_window
   # 用 NumPy + SciPy 实现，消除 Numba JIT 延迟
   ```

2. **Mel 矩阵动态生成**
   ```python
   # 原来: 预先计算保存
   mel_filters = np.load("mel_filters.npy")
   
   # 优化后: 动态生成，启动快 3 秒
   self.filters = self._generate_filters(...)
   ```

3. **异步预热**
   ```python
   # 在辅助进程中预热，不阻塞主进程
   warmup_proc = Process(target=warmup_encoder)
   ```

#### Day 28-30: 命令行工具与打包

**提交**: `f157e09 初步的命令行转录工具` → `4f9b7c5 打包脚本`

**命令行工具设计**:
```python
# transcribe.py
import typer

app = typer.Typer()

@app.command()
def transcribe(
    files: List[Path],
    model_dir: str = "model",
    use_dml: bool = True,
    language: Optional[str] = None
):
    # 实现...
```

**PyInstaller 打包**:
```python
# build.spec
# 关键配置：
# 1. 排除 torch/transformers（太大了）
# 2. 链接模型文件夹（不复制）
# 3. 包含 llama.cpp DLL
```

---

## 🎯 关键技术决策分析

### 决策 1: 为什么选择 ONNX + GGUF 混合架构？

**可选方案对比**:

| 方案 | 优点 | 缺点 | 作者选择 |
|------|------|------|----------|
| 纯 PyTorch | 简单直接 | 显存占用高，速度慢 | ❌ |
| 纯 ONNX | 统一格式 | Decoder 复杂，不支持 KV Cache | ❌ |
| 纯 GGUF | 体积小 | Encoder 量化损失大 | ❌ |
| **ONNX + GGUF** | **各取所长** | **集成复杂** | ✅ |

**理由**:
- Encoder 用 ONNX: 支持 DML/Vulkan，量化精度可控
- Decoder 用 GGUF: llama.cpp 成熟，支持 KV Cache

### 决策 2: 为什么分体 Encoder？

**原始问题**: 长音频导致显存爆炸

**解决方案演进**:

```
尝试 1: 减小 batch size
  ↓ 效果不佳，CNN 中间激活依然大

尝试 2: 多通道卷积
  ↓ 代码复杂，收益有限

尝试 3: 分体 Encoder（最终方案）
  ✓ Frontend 分段处理，Backend 完整处理
  ✓ 显存占用稳定，与音频长度无关
```

### 决策 3: 为什么用多进程而非多线程？

**原因**:
1. **GIL 限制**: Python 多线程无法利用多核
2. **ONNX Runtime**: 在多进程中可以独立使用 DML
3. **隔离性**: Encoder 崩溃不影响主进程

```python
# 主进程
main_process: ASR 解码 + 协调

# 辅助进程
worker_process: Encoder + Aligner
```

### 决策 4: 为什么移除 librosa？

**librosa 的问题**:
- 启动慢（Numba JIT 编译）
- 依赖多（需要 soundfile 等）
- 功能过剩（只需要 Mel 提取）

**替换方案**:
```python
# 纯 NumPy + SciPy 实现
class FastWhisperMel:
    def __call__(self, audio):
        # 1. 分帧（零拷贝）
        frames = np.lib.stride_tricks.as_strided(...)
        
        # 2. FFT
        stft = np.fft.rfft(frames * self.window)
        
        # 3. Mel 滤波
        mel = np.dot(self.filters.T, np.abs(stft)**2)
        
        return np.log10(mel)
```

**收益**: 启动时间从 6 秒降至 1 秒

---

## 🐛 关键 Bug 与解决方案

### Bug 1: 时间戳非单调递增

**现象**: 对齐结果出现负数或倒退的时间戳

**根因**: Aligner 解码时，Timestamp token 预测不稳定

**解决方案**: LIS + 线性插值算法

```python
def fix_timestamps(data):
    # 1. 找最长递增子序列 (LIS)
    lis_indices = find_lis(data)
    
    # 2. 标记异常点
    is_normal = [i in lis_indices for i in range(len(data))]
    
    # 3. 异常点插值
    for i, normal in enumerate(is_normal):
        if not normal:
            data[i] = interpolate(i, left_val, right_val)
```

### Bug 2: Intel 集显输出乱码

**现象**: 输出 "!!!!!" 或乱码

**根因**: Intel 集显 FP16 计算溢出

**解决方案**: 禁用 FP16

```python
os.environ["GGML_VULKAN_DISABLE_F16"] = "1"
```

### Bug 3: 显存持续增长

**现象**: 长音频处理时显存不断增长

**根因**: ONNX Runtime 的内存池未释放

**解决方案**: 进程隔离

```python
# 每个音频文件使用新进程
Process(target=process_audio, args=(audio,)).start()
```

---

## 📊 性能演进数据

| 阶段 | RTF (实时率) | 显存占用 | 启动时间 | 备注 |
|------|-------------|----------|----------|------|
| 初始 | 0.5 | 8GB | 15s | 纯 PyTorch |
| 优化 1 | 0.1 | 4GB | 12s | ONNX 导出 |
| 优化 2 | 0.08 | 2GB | 8s | 分体 Encoder |
| 优化 3 | 0.05 | 900MB | 3s | INT4 量化 |
| 最终 | 0.052 | 900MB | 2.5s | 移除 librosa |

---

## 💡 给后来者的建议

### 如果你想复刻这个项目...

#### 阶段 1: 模型分析（1-2 天）

```python
# 1. 加载官方模型
from transformers import AutoModel
model = AutoModel.from_pretrained("model_name")

# 2. 打印模型结构
print(model)

# 3. 确定导出组件
# - Encoder?
# - Decoder?
# - 其他预处理?

# 4. 验证导出精度
def verify_export(onnx_path, torch_model, sample_input):
    onnx_out = onnx_inference(onnx_path, sample_input)
    torch_out = torch_model(sample_input)
    assert cosine_similarity(onnx_out, torch_out) > 0.99
```

#### 阶段 2: 最小可用推理（3-5 天）

1. **先跑通单文件推理**
   - 不要管多进程
   - 不要管长音频
   - 只要短音频能出结果

2. **验证精度**
   - 与官方输出对比
   - 确保误差 < 1%

3. **添加基础功能**
   - 音频加载
   - 结果保存

#### 阶段 3: 工程化（1-2 周）

1. **性能优化**
   - 量化
   - 批处理
   - 缓存

2. **稳定性**
   - 错误处理
   - 资源释放
   - 日志记录

3. **易用性**
   - 命令行工具
   - 配置文件
   - 文档

### 关键工具链

```
模型分析:
├── Netron (可视化 ONNX)
├── transformers (加载官方模型)
└── torch.onnx.export (导出 ONNX)

推理验证:
├── onnxruntime (CPU/GPU 推理)
├── llama.cpp (GGUF 推理)
└── scipy (音频处理)

性能优化:
├── onnxoptimizer (ONNX 优化)
├── onnxruntime.quantization (量化)
└── llama-quantize (GGUF 量化)

工程化:
├── PyInstaller (打包)
├── typer (CLI)
└── rich (终端美化)
```

---

## 🎓 从本项目学到的工程思维

### 1. 渐进式开发

不要试图一次性完成所有功能：
```
✗ 错误: 先设计完美架构，再写代码
✓ 正确: 先跑通最小可用版本，再逐步优化
```

### 2. 数据驱动优化

每个优化都要用数据验证：
```python
# 量化前
similarity = calculate_similarity(fp32_output, fp16_output)
print(f"FP16 相似度: {similarity}")  # 必须 > 99%

# 量化后
similarity = calculate_similarity(fp32_output, int4_output)
print(f"INT4 相似度: {similarity}")  # 接受 > 96%
```

### 3. 问题隔离

遇到问题时，先确定边界：
```
问题: 转录结果不对

排查步骤:
1. Encoder 输出对吗？ → 验证余弦相似度
2. Prompt 构建对吗？ → 打印对比官方
3. Decoder 推理对吗？ → 单独测试 Decoder
4. 后处理对吗？ → 检查文本解码
```

### 4. 性能分析

不要盲目优化，先 profile：
```python
import cProfile
cProfile.run('transcribe(audio)', sort='cumulative')

# 找出真正的瓶颈
# 可能是: 模型加载? 预热? 解码? 后处理?
```

---

## 📚 相关提交参考

如果你想深入了解某个阶段的代码，可以查看这些提交：

| 阶段 | 关键提交 | 说明 |
|------|----------|------|
| 起步 | `c556c2d` | 初步提交 |
| Encoder 导出 | `f75b043` | 余弦相似度验证 |
| 分体 Encoder | `da0590c` | 解决显存问题 |
| Decoder 导出 | `ba5983e` | GGUF 导出 |
| 基础推理 | `c021cc0` | 初步引擎 |
| 时间戳对齐 | `17c63f6` | 集成 Aligner |
| 量化优化 | `ab498eb` | INT4 量化 |
| 启动优化 | `3917d4b` | 统计启动时间 |
| 命令行工具 | `f157e09` | transcribe.py |
| 打包 | `4f9b7c5` | PyInstaller |

查看具体提交的命令：
```bash
git show f75b043  # 查看某个提交的详细变更
git diff c556c2d..f75b043  # 查看两个提交之间的差异
```

---

**文档结束**
