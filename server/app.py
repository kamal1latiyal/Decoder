"""
FastAPI WebSocket TTS server backed by the megakernel pipeline.

Endpoints:
  GET  /health      → warm status, CUDA device info
  GET  /metrics     → last synthesis metrics (TTFC, RTF, tok/s)
  WS   /synthesize  → streaming TTS: JSON in, binary PCM + JSON done frame out

Run:
  python -m server.app --host 0.0.0.0 --port 8765
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Optional

# uvloop = faster asyncio event loop (~2-4× lower scheduling cost). Drop-in.
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from tts.pipeline import TTSPipeline, SynthesisMetrics

# Backend + reference-voice selection comes in via env so the FastAPI startup
# event can read it without parsing argv twice.
_BACKEND = os.environ.get("TTS_BACKEND", "megakernel")
_REF_AUDIO = os.environ.get("TTS_REF_AUDIO") or None
_REF_TEXT = os.environ.get("TTS_REF_TEXT") or None
# CUDA-graphed subtalker is enabled by default for the megakernel backend.
# Disable via env var or `--no-cuda-graph` to A/B against the HF reference.
_USE_CUDA_GRAPH = os.environ.get("TTS_USE_CUDA_GRAPH", "1") not in ("0", "false", "False")
# Codec chunking knobs — lower chunk_frames cuts TTFC, raise overlap_frames
# to maintain quality. Defaults match the pre-existing 4/4 behaviour.
_CHUNK_FRAMES = int(os.environ.get("TTS_CHUNK_FRAMES", "4"))
_OVERLAP_FRAMES = int(os.environ.get("TTS_OVERLAP_FRAMES", "4"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-TTS Megakernel Server")

# Single shared pipeline instance — warm on startup, GPU is single-tenanted.
_pipeline: Optional[TTSPipeline] = None
_pipeline_lock = asyncio.Lock()
_last_metrics: Optional[SynthesisMetrics] = None
_startup_time: Optional[float] = None


@app.on_event("startup")
async def load_pipeline():
    global _pipeline, _startup_time
    t0 = time.perf_counter()
    log.info(f"Loading TTS pipeline (backend={_BACKEND})...")
    loop = asyncio.get_event_loop()

    def _build():
        return TTSPipeline(
            backend=_BACKEND,
            ref_audio_path=_REF_AUDIO,
            ref_text=_REF_TEXT,
            use_cuda_graph=_USE_CUDA_GRAPH,
            chunk_frames=_CHUNK_FRAMES,
            overlap_frames=_OVERLAP_FRAMES,
        )

    _pipeline = await loop.run_in_executor(None, _build)
    _startup_time = time.perf_counter() - t0
    log.info(f"Pipeline ready in {_startup_time:.1f}s")


@app.get("/health")
async def health():
    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    return JSONResponse({
        "status": "ready" if _pipeline is not None else "loading",
        "cuda_device": device_name,
        "cuda_memory_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if cuda_ok else 0,
        "startup_time_s": round(_startup_time or 0, 2),
    })


@app.get("/metrics")
async def metrics():
    if _last_metrics is None:
        return JSONResponse({"status": "no_requests_yet"})
    m = _last_metrics
    return JSONResponse({
        "ttfc_ms":        round(m.ttfc_ms, 2),
        "rtf":            round(m.rtf, 4),
        "tokens_per_sec": round(m.tokens_per_sec, 1),
        "total_tokens":   m.total_tokens,
        "audio_duration_s": round(m.audio_duration_s, 3),
        "wall_time_s":    round(m.wall_time_s, 3),
    })


@app.websocket("/synthesize")
async def synthesize(ws: WebSocket):
    """
    WebSocket endpoint for streaming TTS synthesis.

    Protocol:
      Client → Server:  JSON {"text": "...", "speaker": "default"}
      Server → Client:  binary frames — raw int16 PCM, 24 kHz mono
      Server → Client:  JSON {"type": "done", "metrics": {...}}
    """
    global _last_metrics

    await ws.accept()
    log.info(f"New connection from {ws.client}")

    if _pipeline is None:
        await ws.send_text(json.dumps({"type": "error", "message": "Pipeline not ready"}))
        await ws.close()
        return

    try:
        raw = await ws.receive_text()
        request = json.loads(raw)
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": f"Bad request: {e}"}))
        await ws.close()
        return

    text = request.get("text", "").strip()
    speaker = request.get("speaker", "default")

    if not text:
        await ws.send_text(json.dumps({"type": "error", "message": "Empty text"}))
        await ws.close()
        return

    log.info(f"Synthesizing [{len(text)} chars]: {text[:60]}...")

    # Acquire the lock — GPU is single-tenanted, queue concurrent requests
    async with _pipeline_lock:
        try:
            chunks_sent = 0
            async for pcm_bytes in _pipeline.synthesize(text, speaker=speaker):
                await ws.send_bytes(pcm_bytes)
                chunks_sent += 1

            _last_metrics = _pipeline.last_metrics
            m = _last_metrics
            log.info(
                f"Done: {chunks_sent} chunks, TTFC={m.ttfc_ms:.1f}ms "
                f"RTF={m.rtf:.3f} tok/s={m.tokens_per_sec:.0f}"
            )
            await ws.send_text(json.dumps({
                "type": "done",
                "metrics": {
                    "ttfc_ms":        round(m.ttfc_ms, 2),
                    "rtf":            round(m.rtf, 4),
                    "tokens_per_sec": round(m.tokens_per_sec, 1),
                    "total_tokens":   m.total_tokens,
                    "audio_duration_s": round(m.audio_duration_s, 3),
                },
            }))

        except WebSocketDisconnect:
            log.info("Client disconnected mid-stream")
        except Exception as e:
            log.exception(f"Synthesis error: {e}")
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Megakernel WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--backend", choices=("megakernel", "hf"), default=_BACKEND,
                        help="Talker decode backend (megakernel for kernel decode, hf for baseline).")
    parser.add_argument("--ref-audio", default=_REF_AUDIO,
                        help="Path to reference wav for voice cloning. If omitted, x-vector default voice is used.")
    parser.add_argument("--ref-text", default=_REF_TEXT,
                        help="Transcript of the reference audio (required for ICL mode).")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="Disable the CUDA-graphed subtalker (A/B against HF reference).")
    parser.add_argument("--chunk-frames", type=int, default=_CHUNK_FRAMES,
                        help=("How many new codec frames per decode call. Default 4. "
                              "Lower = lower TTFC, but more codec calls per second of audio. "
                              "Try 1-2 with --overlap-frames 6-8 for aggressive low-latency."))
    parser.add_argument("--overlap-frames", type=int, default=_OVERLAP_FRAMES,
                        help="Causal-context frames per codec call. Default 4. Raise when --chunk-frames is small.")
    args = parser.parse_args()

    # Propagate to the startup event via env (uvicorn re-imports the module).
    os.environ["TTS_BACKEND"] = args.backend
    if args.ref_audio:
        os.environ["TTS_REF_AUDIO"] = args.ref_audio
    if args.ref_text:
        os.environ["TTS_REF_TEXT"] = args.ref_text
    if args.no_cuda_graph:
        os.environ["TTS_USE_CUDA_GRAPH"] = "0"
    os.environ["TTS_CHUNK_FRAMES"] = str(args.chunk_frames)
    os.environ["TTS_OVERLAP_FRAMES"] = str(args.overlap_frames)

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        ws_ping_interval=None,   # disable WS keepalive pings during long synthesis
    )


if __name__ == "__main__":
    main()
