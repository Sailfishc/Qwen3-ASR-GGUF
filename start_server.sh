#!/bin/zsh
# 启动 Qwen3-ASR OpenAI 兼容转录服务
cd "$(dirname "$0")"
python3.11 serve_openai_gguf.py --model-dir ./model --port 8001
