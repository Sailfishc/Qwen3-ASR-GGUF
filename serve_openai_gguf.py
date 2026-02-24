# coding=utf-8
"""
OpenAI-compatible audio transcription API server backed by QwenASREngine (GGUF).

Usage:
    pip install fastapi uvicorn python-multipart
    python serve_openai_gguf.py --model-dir ./model --port 8000

Then point any OpenAI-compatible client (e.g. Spokenly) at:
    http://localhost:8000
"""
import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, AlignerConfig
from qwen_asr_gguf.inference.exporters import alignment_to_srt, alignment_to_json

# ---------------------------------------------------------------------------
# Language mapping: ISO-639-1 code → engine language name
# ---------------------------------------------------------------------------
LANGUAGE_MAP = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese",
    "ar": "Arabic", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "id": "Indonesian",
    "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
    "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "tl": "Filipino", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-ASR OpenAI-compatible API")
engine: Optional[QwenASREngine] = None
args: Optional[argparse.Namespace] = None

# In-memory stats
_stats = {
    "total": 0,
    "success": 0,
    "error": 0,
    "total_wall_time": 0.0,   # 服务端总耗时（秒）
    "history": [],             # 最近 50 条记录
}


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global engine, args
    config = ASREngineConfig(
        model_dir=args.model_dir,
        use_dml=args.dml,
        n_ctx=args.n_ctx,
        chunk_size=float(args.chunk_size),
        enable_aligner=args.enable_aligner,
        verbose=False,
    )
    engine = QwenASREngine(config)
    print(f"[serve] Engine ready. Listening on {args.host}:{args.port}")


@app.on_event("shutdown")
async def shutdown():
    global engine
    if engine is not None:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _resolve_language(lang_code: Optional[str]) -> Optional[str]:
    """Convert ISO-639-1 code to engine language name.
    Unknown codes are passed through unchanged so the engine can handle them.
    """
    if not lang_code:
        return None
    return LANGUAGE_MAP.get(lang_code.lower(), lang_code)


def _map_temperature(t: float) -> float:
    """OpenAI sends 0.0 for greedy; map that to the engine's recommended 0.4."""
    return 0.4 if t == 0.0 else t


def _srt_to_vtt(srt_text: str) -> str:
    """Minimal SRT → WebVTT conversion (header + comma→dot in timestamps)."""
    import re
    vtt = "WEBVTT\n\n"
    # Replace comma separators in timestamps: 00:00:01,234 → 00:00:01.234
    vtt += re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", srt_text)
    return vtt


def _make_verbose_json(result, language_name: Optional[str], want_words: bool) -> dict:
    """Build the verbose_json response payload."""
    words: List[dict] = []
    segments: List[dict] = []

    if result.alignment and want_words:
        words = alignment_to_json(result.alignment.items)

    # Build coarse segments from the text (split on sentence boundaries).
    # This is a best-effort approximation when alignment data is unavailable.
    if result.alignment:
        # Use alignment items to form segments grouped by punctuation
        items = result.alignment.items
        seg_buf = []
        seg_start = None
        for item in items:
            if seg_start is None:
                seg_start = item.start_time
            seg_buf.append(item.text)
            combined = "".join(seg_buf)
            if any(combined.endswith(p) for p in "，。？！,.?!\n") or len(combined) >= 40:
                segments.append({
                    "start": round(seg_start, 3),
                    "end": round(item.end_time, 3),
                    "text": combined.strip(),
                })
                seg_buf = []
                seg_start = None
        if seg_buf:
            segments.append({
                "start": round(seg_start, 3),
                "end": round(items[-1].end_time, 3),
                "text": "".join(seg_buf).strip(),
            })
    else:
        # No alignment: single segment with no timestamps
        segments = [{"start": 0.0, "end": 0.0, "text": result.text.strip()}]

    duration = 0.0
    if result.alignment and result.alignment.items:
        duration = result.alignment.items[-1].end_time

    return {
        "task": "transcribe",
        "language": language_name or "unknown",
        "duration": round(duration, 3),
        "text": result.text,
        "words": words,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "engine_ready": engine is not None}


@app.get("/stats")
async def stats():
    """请求统计：总量、成功率、平均耗时、最近记录。"""
    total = _stats["total"]
    success_rate = (_stats["success"] / total * 100) if total > 0 else 0.0
    avg_time = (_stats["total_wall_time"] / _stats["success"]) if _stats["success"] > 0 else 0.0
    return {
        "total_requests": total,
        "success": _stats["success"],
        "error": _stats["error"],
        "success_rate_pct": round(success_rate, 1),
        "avg_wall_time_sec": round(avg_time, 2),
        "recent": _stats["history"][-20:],  # 最近 20 条
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "owned_by": "qwen3-asr-gguf",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: List[str] = Form([]),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    # Determine whether word-level timestamps are requested
    want_words = "word" in timestamp_granularities

    # Resolve language
    engine_language = _resolve_language(language)
    engine_temp = _map_temperature(temperature)

    # Save upload to a temp file (preserve original suffix for format detection)
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 读取音频时长（用于计算 RTF），在转录前从文件读取
    audio_dur = None
    try:
        from pydub import AudioSegment
        audio_dur = len(AudioSegment.from_file(tmp_path)) / 1000.0
    except Exception:
        pass

    _stats["total"] += 1
    t0 = time.time()
    record = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "file": file.filename or "unknown",
        "language": language or "auto",
        "ok": False,
        "wall_sec": 0.0,
        "rtf": None,
        "chars": 0,
    }

    try:
        result = engine.transcribe(
            audio_file=tmp_path,
            language=engine_language,
            context=prompt or "",
            temperature=engine_temp,
        )
        wall = time.time() - t0
        _stats["success"] += 1
        _stats["total_wall_time"] += wall

        # 优先用 aligner 结果的精确时长，其次用文件读取的时长
        if result.alignment and result.alignment.items:
            audio_dur = result.alignment.items[-1].end_time

        record.update({
            "ok": True,
            "wall_sec": round(wall, 2),
            "rtf": round(wall / audio_dur, 3) if audio_dur else None,
            "chars": len(result.text),
        })
        print(
            f"[transcribe] ✓ {file.filename!r}  {wall:.1f}s"
            + (f"  RTF={record['rtf']}" if record["rtf"] else "")
            + f"  {len(result.text)} chars"
            + f"\n             → {result.text}"
        )
    except Exception as exc:
        _stats["error"] += 1
        record["wall_sec"] = round(time.time() - t0, 2)
        print(f"[transcribe] ✗ {file.filename!r}  {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _stats["history"].append(record)
        if len(_stats["history"]) > 50:
            _stats["history"].pop(0)
        os.unlink(tmp_path)

    # Format response
    fmt = response_format.lower()

    if fmt == "text":
        return PlainTextResponse(result.text)

    if fmt == "srt":
        if result.alignment:
            srt_content = alignment_to_srt(result.alignment.items)
        else:
            srt_content = ""
        return PlainTextResponse(srt_content, media_type="text/plain")

    if fmt == "vtt":
        if result.alignment:
            srt_content = alignment_to_srt(result.alignment.items)
            vtt_content = _srt_to_vtt(srt_content)
        else:
            vtt_content = "WEBVTT\n\n"
        return PlainTextResponse(vtt_content, media_type="text/vtt")

    if fmt == "verbose_json":
        payload = _make_verbose_json(result, engine_language, want_words)
        return JSONResponse(payload)

    # Default: json
    return JSONResponse({"text": result.text})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Qwen3-ASR as an OpenAI-compatible transcription API"
    )
    parser.add_argument("--model-dir", default="./model", help="Model weights directory (default: ./model)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument("--enable-aligner", action="store_true", help="Enable forced aligner for word-level timestamps")
    parser.add_argument("--dml", action="store_true", help="Enable DirectML acceleration (Windows)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="LLM context size (default: 2048)")
    parser.add_argument("--chunk-size", type=float, default=40.0, help="Audio chunk size in seconds (default: 40)")
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
