"""
Full Pipecat voice pipeline demo: STT → LLM → Megakernel TTS → audio output.

Pipeline:
  Microphone (WebSocket transport)
    → Deepgram STT
    → Claude Opus 4.7 (claude-opus-4-7)
    → MegakernelTTSService
    → Speaker (WebSocket transport)

Run (after starting the TTS server):
  export DEEPGRAM_API_KEY=...
  export ANTHROPIC_API_KEY=...
  python pipecat_integration/demo.py

For a local audio demo (no WebSocket client needed):
  python pipecat_integration/demo.py --local
"""

import argparse
import asyncio
import logging
import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

from pipecat_integration.tts_service import MegakernelTTSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful voice assistant powered by a custom CUDA-accelerated TTS system. "
    "Keep responses concise and conversational — one to three sentences. "
    "You are running on an RTX 5090 with a custom megakernel for ultra-low-latency speech."
)

TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "ws://localhost:8765")


async def run_pipeline(ws_host: str = "0.0.0.0", ws_port: int = 8766):
    """Run the voice pipeline, accepting audio over WebSocket."""

    transport = WebsocketServerTransport(
        host=ws_host,
        port=ws_port,
        params=WebsocketServerParams(
            add_wav_header=False,   # raw PCM
            audio_out_sample_rate=24000,
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        audio_encoding="linear16",
        sample_rate=16000,
    )

    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-opus-4-7",
    )

    tts = MegakernelTTSService(
        server_url=TTS_SERVER_URL,
        speaker="default",
    )

    context = LLMContext()
    context.add_message({"role": "system", "content": SYSTEM_PROMPT})

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline([
        transport.input(),    # InputAudioRawFrame from WebSocket client
        stt,                  # → TranscriptionFrame
        user_agg,             # VAD + context accumulation
        llm,                  # → LLMTextFrame stream
        tts,                  # → TTSAudioRawFrame stream
        transport.output(),   # PCM audio → WebSocket client
        assistant_agg,        # accumulate assistant text into context
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    log.info(f"Voice pipeline listening on ws://{ws_host}:{ws_port}")
    log.info(f"TTS server: {TTS_SERVER_URL}")
    log.info("Connect with a WebSocket client sending 16kHz mono int16 PCM audio.")

    runner = PipelineRunner()
    await runner.run(task)


def _load_dotenv_if_present():
    """Read KEY=VALUE lines from ./.env into os.environ (does not overwrite existing).
    No dependency on python-dotenv; demo should run on any fresh env."""
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def main():
    parser = argparse.ArgumentParser(description="Pipecat Megakernel TTS Demo")
    parser.add_argument("--ws-host", default="0.0.0.0", help="WebSocket listen host")
    parser.add_argument("--ws-port", type=int, default=8766, help="WebSocket listen port")
    args = parser.parse_args()

    _load_dotenv_if_present()

    missing = [k for k in ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY") if k not in os.environ]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
        print("  Set them in your shell, or copy .env.example → .env and fill in.",
              file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_pipeline(ws_host=args.ws_host, ws_port=args.ws_port))


if __name__ == "__main__":
    main()
