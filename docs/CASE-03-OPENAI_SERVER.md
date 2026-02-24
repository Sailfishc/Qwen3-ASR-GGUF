# CASE-03：OpenAI 兼容转录 API 服务

将 Qwen3-ASR GGUF 引擎包装为 HTTP 服务，任何支持 OpenAI Whisper API 的客户端（Spokenly、自定义脚本等）都可以直接接入。

---

## 一、安装依赖

首次使用前执行一次：

```bash
pip3 install fastapi uvicorn python-multipart
```

---

## 二、快速启动

```bash
./start_server.sh
```

或手动指定参数：

```bash
python3.11 serve_openai_gguf.py --model-dir ./model --port 8001
```

启动成功后终端显示：

```
[serve] Engine ready. Listening on 127.0.0.1:8001
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

按 `Ctrl+C` 关闭服务。

---

## 三、启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | `./model` | 模型文件目录 |
| `--host` | `127.0.0.1` | 监听地址（改为 `0.0.0.0` 可局域网访问） |
| `--port` | `8001` | 监听端口 |
| `--enable-aligner` | 关闭 | 启用字级时间戳对齐（需要 aligner 模型文件） |
| `--n-ctx` | `2048` | LLM 上下文大小 |
| `--chunk-size` | `40` | 音频分段时长（秒） |
| `--dml` | 关闭 | 启用 DirectML 加速（仅 Windows） |

---

## 四、API 接口

### 4.1 转录音频 `POST /v1/audio/transcriptions`

与 OpenAI Whisper API 兼容，multipart/form-data 格式。

**请求字段：**

| 字段 | 必填 | 说明 |
|------|------|------|
| `file` | 是 | 音频文件（wav / mp3 / m4a / flac 等） |
| `model` | 否 | 填任意值，如 `whisper-1`（不影响实际模型） |
| `language` | 否 | ISO-639-1 语言代码，如 `zh`、`en`。不填则自动识别 |
| `prompt` | 否 | 上下文提示，可提升专有名词识别率 |
| `response_format` | 否 | 返回格式，默认 `json`，见下表 |
| `temperature` | 否 | 采样温度 0~1，默认 `0`（映射为引擎的 0.4） |

**response_format 选项：**

| 值 | 返回内容 |
|----|---------|
| `json`（默认） | `{"text": "转录结果"}` |
| `text` | 纯文本字符串 |
| `verbose_json` | 含时长、分段、词级时间戳的完整 JSON |
| `srt` | SRT 字幕格式（需启用 aligner） |
| `vtt` | WebVTT 字幕格式（需启用 aligner） |

**curl 示例：**

```bash
# 基本转录
curl -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"

# 指定语言 + 上下文提示
curl -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@meeting.mp3" \
  -F "model=whisper-1" \
  -F "language=zh" \
  -F "prompt=本次会议讨论产品路线图" \
  -F "response_format=json"

# 返回纯文本
curl -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=text"
```

---

### 4.2 查看请求统计 `GET /stats`

```bash
curl http://127.0.0.1:8001/stats
```

返回示例：

```json
{
  "total_requests": 12,
  "success": 11,
  "error": 1,
  "success_rate_pct": 91.7,
  "avg_wall_time_sec": 3.42,
  "recent": [
    {
      "time": "14:32:01",
      "file": "meeting.mp3",
      "language": "zh",
      "ok": true,
      "wall_sec": 3.2,
      "rtf": 0.085,
      "chars": 142
    }
  ]
}
```

**字段说明：**

| 字段 | 含义 |
|------|------|
| `wall_sec` | 服务端总耗时（秒），包含音频读取 + 推理 |
| `rtf` | 实时率 = 耗时 / 音频时长，越小越快；< 1 表示快于实时 |
| `chars` | 转录文本字符数 |

---

### 4.3 其他接口

```bash
# 健康检查
curl http://127.0.0.1:8001/health

# 查看可用模型列表
curl http://127.0.0.1:8001/v1/models
```

---

## 五、客户端接入

### Spokenly

1. 打开 Spokenly → 设置 → 转录引擎
2. 选择 **OpenAI Compatible API**
3. 填写：
   - **API Base URL**：`http://127.0.0.1:8001`
   - **API Key**：任意字符串（服务不验证）
   - **Model**：`whisper-1`

### Python 脚本

```python
from openai import OpenAI

client = OpenAI(
    api_key="any",
    base_url="http://127.0.0.1:8001/v1"
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        language="zh",
    )
print(result.text)
```

---

## 六、语言代码对照

服务接受 ISO-639-1 两字母代码，自动映射为引擎语言名：

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `zh` | 中文 | `en` | 英文 |
| `yue` | 粤语 | `ja` | 日文 |
| `ko` | 韩文 | `fr` | 法文 |
| `de` | 德文 | `es` | 西班牙文 |
| `ru` | 俄文 | `ar` | 阿拉伯文 |

不填 `language` 则自动识别语种。

---

## 七、局域网共享

默认只监听本机（`127.0.0.1`）。若要让同局域网其他设备访问：

```bash
python3.11 serve_openai_gguf.py --model-dir ./model --host 0.0.0.0 --port 8001
```

其他设备将 URL 中的 `127.0.0.1` 替换为本机 IP 即可。

---

## 八、终端实时日志

每次转录完成，终端打印一行摘要：

```
[transcribe] ✓ 'audio.wav'  3.2s  RTF=0.085  142 chars
[transcribe] ✗ 'bad.xyz'  0.1s  <错误信息>
```
