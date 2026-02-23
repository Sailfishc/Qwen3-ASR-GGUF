# 实战案例：在 macOS 上跑通 GGUF 推理路径

> 本文是 [PyTorch 路径实战](./PRACTICAL_CASE_PYTORCH_INFERENCE.md) 的续篇。
> 我们已经用 PyTorch 路径成功转录了音频，本文记录如何走通
> **这个项目真正的核心路径**——GGUF 推理——以及途中踩过的每一个坑。

---

## 背景与目标

上一篇文档已经：

- 把 `test_audio.wav`（16 秒，16kHz 单声道）准备就绪
- 安装了 torch、onnxruntime、transformers==4.57.6 等依赖
- 用 PyTorch 路径完成了转录，RTF ≈ 0.43

**本篇目标**：改用 `transcribe.py`（GGUF 路径）转录同一段音频，
体验 GGUF 方案在速度上的实际差距。

GGUF 路径需要两样东西：

```
需求 1：model/ 目录           需求 2：qwen_asr_gguf/inference/bin/
├── qwen3_asr_llm.q4_k.gguf   ├── libllama.dylib
├── qwen3_asr_encoder_frontend.int4.onnx
└── qwen3_asr_encoder_backend.int4.onnx
```

---

## 第一步：排查已有条件

```bash
ls model/          # → 不存在
ls qwen_asr_gguf/inference/bin/   # → 不存在
```

两样东西都缺，需要从头获取。

---

## 第二步：下载 GGUF 模型文件

项目在 GitHub Releases 发布了两个 tag：

```bash
gh release list --repo HaujetZhao/Qwen3-ASR-GGUF

# 输出：
# Qwen3-ASR-Transcribe 转录工具  Latest  v0.1   2026-02-22  (Windows 可执行文件)
# GGUF 模型下载                          models  2026-02-21  (模型文件)
```

`models` tag 下有：
- `Qwen3-ASR-0.6B-gguf.zip`（538MB）
- `Qwen3-ASR-1.7B-gguf.zip`
- `Qwen3-ForceAligner-0.6B-gguf.zip`

下载 0.6B 模型包：

```bash
gh release download models --repo HaujetZhao/Qwen3-ASR-GGUF \
    --pattern "Qwen3-ASR-0.6B-gguf.zip" \
    --dir /tmp/gguf_dl \
    --clobber
```

> **踩坑**：第一次下载得到了 111MB 的文件，直接解压报错
> `End-of-central-directory signature not found`（文件损坏）。
> 原因：`gh release download` 命令在后台运行时被提前中断，文件未下载完整。
> **解决**：前台同步执行，等待完整下载（真实大小 538MB）再解压。

解压并放入 `model/` 目录：

```bash
mkdir -p model
unzip /tmp/gguf_dl/Qwen3-ASR-0.6B-gguf.zip -d /tmp/gguf_extract/
cp /tmp/gguf_extract/*.onnx /tmp/gguf_extract/*.gguf ./model/
```

解压后的文件：

```
model/
├── qwen3_asr_encoder_backend.int4.onnx   90 MB   (Encoder 后端，Transformer 层)
├── qwen3_asr_encoder_frontend.int4.onnx  19 MB   (Encoder 前端，CNN 层)
└── qwen3_asr_llm.q4_k.gguf             462 MB   (Decoder，q4_k 量化)
```

> **与 PyTorch 模型的大小对比**：
> PyTorch 原始权重 1.8 GB (fp32) → GGUF 套件合计 571 MB（int4量化）
> 体积压缩到约 **32%**，同时量化损失极小（困惑度仅增加 8.7%）。

---

## 第三步：获取预编译的 libllama.dylib

GGUF 路径的 Decoder 通过 `llama.py` 用 ctypes 直接调用 `libllama.dylib`。
这个动态库需要手动提供，项目没有打包进源码（仅在 Windows Release 包里有）。

### 方案 A：从 llama-cpp-python 借用（失败）

最简单的方案是安装 `llama-cpp-python`，它在安装时会编译并捆绑 `libllama.dylib`：

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip3 install llama-cpp-python --break-system-packages
```

安装完成后，在以下路径找到所有 dylib：

```
/opt/homebrew/lib/python3.11/site-packages/llama_cpp/lib/
├── libllama.dylib
├── libggml.dylib
├── libggml-base.dylib
├── libggml-blas.dylib
├── libggml-cpu.dylib
└── libggml-metal.dylib
```

复制到项目的 `bin/` 目录：

```bash
mkdir -p qwen_asr_gguf/inference/bin
cp /opt/homebrew/lib/python3.11/site-packages/llama_cpp/lib/lib*.dylib \
   qwen_asr_gguf/inference/bin/
```

运行时第一次报错（缺 libggml-blas.dylib）：

```
dlopen(libggml.dylib): Library not loaded: @rpath/libggml-blas.dylib
```

补充复制 `libggml-blas.dylib` 后，第二次出现 segfault（exit code 139）：

```
--- [QwenASR] 初始化引擎 (DML: False) ---
[进程崩溃，无输出]
```

**原因分析**：`llama-cpp-python 0.3.16` 对应的 llama.cpp 版本较旧，
其 C 结构体布局与项目 `llama.py` 中定义的不一致，导致内存越界崩溃。

具体差异：项目 `llama.py` 的 `llama_context_params` 结构体包含这些新字段：

```python
("flash_attn_type", ctypes.c_int32),   # 枚举类型（旧版是 bool）
("op_offload", ctypes.c_bool),
("swa_full", ctypes.c_bool),           # PR #13194 新增
("kv_unified", ctypes.c_bool),         # PR #14363 新增
("samplers", ctypes.POINTER(...)),      # 新增 sampler 配置
("n_samplers", ctypes.c_size_t),
```

这些字段在 `llama-cpp-python 0.3.16` 所用的 llama.cpp 版本中尚不存在。
**结论**：不能借用旧版本的 dylib，必须从匹配版本的源码编译。

---

### 方案 B：从项目自带的 ref/llama.cpp 编译（部分失败）

项目在 `ref/llama.cpp/` 下保存了对应的 llama.cpp 源码快照，
其 `include/llama.h` 包含了上述所有新字段，版本匹配。

尝试 cmake 配置：

```bash
cmake -S ref/llama.cpp -B /tmp/llama_build \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
```

报错：

```
The source directory does not contain a CMakeLists.txt file.
```

**原因**：`ref/llama.cpp/` 是一份不完整的源码快照——
根目录的 `CMakeLists.txt` 缺失，`tools/` 子目录也不存在。
这份代码可能是通过选择性复制部分文件得到的，并非完整 clone。

尝试从 GitHub 克隆最新 master 的 `CMakeLists.txt` 补入，仍报错：

```
The source directory does not contain a CMakeLists.txt file.
# （tools/ 目录依然缺失，cmake 阶段失败）
```

**结论**：`ref/llama.cpp` 无法直接用于编译，需要完整仓库。

---

### 方案 C：克隆完整 llama.cpp 编译（成功）

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama_full
```

cmake 配置（关闭 BLAS 避免额外依赖，开启 Metal 利用 Apple Silicon）：

```bash
cmake -S /tmp/llama_full -B /tmp/lb3 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DGGML_METAL=ON \
    -DGGML_BLAS=OFF
```

编译（使用所有 CPU 核心，约 2-3 分钟）：

```bash
cmake --build /tmp/lb3 --config Release \
    -j$(sysctl -n hw.logicalcpu) \
    --target llama ggml
```

编译产出：

```
/tmp/lb3/bin/
├── libllama.dylib      2.0 MB
├── libggml.dylib        58 KB
├── libggml-base.dylib  637 KB
├── libggml-cpu.dylib   875 KB
└── libggml-metal.dylib 764 KB
```

复制到项目 `bin/` 目录（覆盖旧文件）：

```bash
cp /tmp/lb3/bin/libllama.dylib \
   /tmp/lb3/bin/libggml.dylib \
   /tmp/lb3/bin/libggml-base.dylib \
   /tmp/lb3/bin/libggml-cpu.dylib \
   /tmp/lb3/bin/libggml-metal.dylib \
   qwen_asr_gguf/inference/bin/
```

---

## 第四步：运行 GGUF 推理

```bash
python3 transcribe.py test_audio.wav \
    --model-dir ./model \
    --prec int4 \
    --no-dml \      # macOS 不支持 DirectML（Windows 专用）
    --no-vulkan \   # 本次不需要 Vulkan 加速
    --no-ts \       # 不启用时间戳对齐（节省时间，先跑通基本路径）
    -y              # 覆盖已存在的输出文件
```

输出：

```
╭──────── Qwen3-ASR 配置选项 ────────╮
│  模型目录    ./model               │
│  编码精度    int4                  │
│  加速设备    DML:OFF | Vulkan:OFF  │
│  时间戳对齐  禁用                  │
│  语言设定    自动识别              │
╰────────────────────────────────────╯
--- [QwenASR] 初始化引擎 (DML: False) ---
--- [QwenASR] 辅助进程已就绪 ---
--- [QwenASR] 引擎初始化耗时: 8.03 秒 ---

开始处理: test_audio.wav

Okay,
 那应该是有已经有了足够的信息，
呃，
为所有端写一份架构文档，
让我快速地理解这个项目。

📊 性能统计:
  🔹 RTF (实时率) : 0.104 (越小越快)
  🔹 音频时长    : 15.96 秒
  🔹 总处理耗时  : 1.66 秒
  🔹 编码等待    : 0.86 秒
  🔹 LLM 预填充  : 0.542 秒 (540 tokens, 995.8 tokens/s)
  🔹 LLM 生成    : 0.251 秒 (29 tokens, 115.7 tokens/s)
✅ 已保存文本文件: test_audio.txt
--- [QwenASR] 引擎已关闭 ---
```

---

## 第五步：对比两条路径的性能

| 指标 | PyTorch 路径 | GGUF 路径 | 差距 |
|------|:-----------:|:---------:|:----:|
| 模型大小 | 1.8 GB (fp32) | 571 MB (int4) | GGUF 小 3.1x |
| 模型加载耗时 | 3.8 秒 | 8.0 秒（含子进程预热）| PyTorch 略快 |
| 转录耗时（16s音频）| 6.8 秒 | **1.66 秒** | **GGUF 快 4.1x** |
| RTF（实时率）| 0.43 | **0.104** | **GGUF 快 4.1x** |
| 依赖 | PyTorch + transformers | onnxruntime + libllama | GGUF 更轻 |
| 跨平台加速 | MPS/CUDA | Metal/DirectML | 不同 |

**备注**：模型加载 GGUF 稍慢，因为它需要额外启动一个子进程（ONNX Encoder Worker）并完成预热（跑一次空推理），这是一次性成本。对于长音频或批量处理场景，这个差距可以忽略不计。

---

## 深入理解：GGUF 引擎启动时发生了什么

`QwenASREngine.__init__()` 做了三件事：

```python
# 1. 启动辅助子进程（运行 ONNX Encoder）
self.helper_proc = mp.Process(
    target=asr_helper_worker_proc,    # → encoder.py
    args=(to_worker_q, from_enc_q, ..., config),
    daemon=True
)
self.helper_proc.start()

# 2. 在主进程加载 GGUF Decoder（通过 ctypes 调用 libllama.dylib）
self.model = llama.LlamaModel(llm_gguf)          # 加载 .gguf 文件
self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
self.ctx = llama.LlamaContext(self.model, n_ctx=2048, ...)

# 3. 等待子进程就绪信号（子进程完成 ONNX 模型加载 + 预热后发出）
msg = self.from_enc_q.get()  # 阻塞等待
```

**为什么 Encoder 要放在子进程？**

- Encoder（ONNX）和 Decoder（llama.cpp）是两个独立的推理框架
- 子进程隔离避免了两个框架之间的线程冲突
- 流式处理时，Encoder 可以提前编码下一段音频，
  和 Decoder 生成文本形成**流水线并行**，减少等待时间
- 通过 `multiprocessing.Queue` 通信，`MSG_EMBD` 消息携带 audio_embedding

**推理阶段的数据流（以 16 秒音频为例）**：

```
主进程                              子进程（Encoder Worker）
  │                                   │
  │─ CMD_ENCODE（发送音频数据）──────→│
  │                                   ├─ ONNX Frontend（CNN） 0.3s
  │                                   ├─ ONNX Backend（Transformer） 0.5s
  │←─ MSG_EMBD（返回 audio_embedding）│  编码等待: 0.86s
  │
  ├─ _build_prompt_embd()
  │   构建: [BOS][system][user][audio_embd][language...][text]
  │
  ├─ llama_decode()（prefill）   0.54s  540 tokens
  │
  └─ llama_decode() × 29次（generate） 0.25s  → 29 tokens
      每次取 logits → 采样 → 取下一个 token
      遇到 </s> 或 <|im_end|> 停止
```

---

## 遇到的坑及解决方案汇总

| 问题 | 现象 | 原因 | 解决 |
|------|------|------|------|
| 下载中断 | unzip 报错「非 zip 文件」 | 后台下载未完成 | 前台同步下载 |
| dylib 缺失 | `Library not loaded: @rpath/libggml-blas.dylib` | 复制 dylib 不完整 | 补充复制所有依赖 |
| API 版本不兼容 | segfault（exit code 139） | llama-cpp-python 版本过旧，struct 布局不同 | 从源码编译匹配版本 |
| ref/llama.cpp 不完整 | cmake 报错「无 CMakeLists.txt」 | 源码目录只是部分快照 | 克隆完整仓库 |
| cmake 缓存冲突 | Re-run cmake with different source | 同一个 build 目录混用了两个 source | 新建 build 目录 |

---

## 附：完整复现命令（从零开始）

```bash
cd /path/to/Qwen3-ASR-GGUF

# ── 步骤 1：安装依赖 ──────────────────────────────────────────────
pip3 install torch torchaudio --break-system-packages
pip3 install onnxruntime librosa pydub srt typer rich \
             nagisa sentencepiece accelerate --break-system-packages
pip3 install "transformers==4.57.6" --break-system-packages

# ── 步骤 2：下载并解压 GGUF 模型 ─────────────────────────────────
mkdir -p model /tmp/gguf_dl
gh release download models --repo HaujetZhao/Qwen3-ASR-GGUF \
    --pattern "Qwen3-ASR-0.6B-gguf.zip" \
    --dir /tmp/gguf_dl --clobber
unzip /tmp/gguf_dl/Qwen3-ASR-0.6B-gguf.zip -d /tmp/gguf_extract/
cp /tmp/gguf_extract/*.onnx /tmp/gguf_extract/*.gguf ./model/

# ── 步骤 3：编译 libllama.dylib ───────────────────────────────────
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama_full
cmake -S /tmp/llama_full -B /tmp/lb \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DGGML_METAL=ON \
    -DGGML_BLAS=OFF
cmake --build /tmp/lb --config Release \
    -j$(sysctl -n hw.logicalcpu) --target llama ggml

mkdir -p qwen_asr_gguf/inference/bin
cp /tmp/lb/bin/libllama.dylib \
   /tmp/lb/bin/libggml.dylib \
   /tmp/lb/bin/libggml-base.dylib \
   /tmp/lb/bin/libggml-cpu.dylib \
   /tmp/lb/bin/libggml-metal.dylib \
   qwen_asr_gguf/inference/bin/

# ── 步骤 4：转录 ────────────────────────────────────────────────
python3 transcribe.py test_audio.wav \
    --model-dir ./model \
    --prec int4 \
    --no-dml \
    --no-vulkan \
    --no-ts \
    -y
```

---

## 后记：ref/llama.cpp 目录的用途

`ref/llama.cpp/` 不是用来编译 dylib 的——它保留的是**头文件和源码参考**，
目的是当项目代码需要对齐新的 llama.cpp API 时，
开发者可以在本地查阅对应版本的 `include/llama.h`，
而不必每次去网上查，也方便 diff 比较 API 变化。

实际的 dylib 在正式发布时会提前编译好，随 Release 包一起分发。
本文中我们在 macOS 上手动编译，正是模拟了这个打包流程。
