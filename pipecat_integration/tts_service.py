"""
MegakernelTTSService — Pipecat TTSService backed by the megakernel WebSocket server.

Subclasses pipecat's TTSService and implements run_tts() as an async generator
that connects to the local WebSocket server and streams PCM chunks as
TTSAudioRawFrame objects.

Usage:
    tts = MegakernelTTSService(server_url="ws://localhost:8765")
    # then wire into Pipeline([..., tts, ...])
"""

import json
import logging
from typing import AsyncGenerator, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
NUM_CHANNELS = 1


class MegakernelTTSService(TTSService):
    """
    Pipecat TTS service that streams audio from the megakernel WebSocket server.

    Connects a fresh WebSocket per synthesis request. Each binary message from
    the server is one PCM chunk (int16 LE, 24kHz mono). Yields TTSAudioRawFrame
    for each chunk — Pipecat dispatches these to the audio output immediately.

    Args:
        server_url: WebSocket URL of the megakernel TTS server.
                    Default: ws://localhost:8765
        speaker:    Speaker identity to pass to the server.
    """

    def __init__(
        self,
        *,
        server_url: str = "ws://localhost:8765",
        speaker: str = "default",
        **kwargs,
    ):
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
        self._server_url = server_url.rstrip("/")
        self._speaker = speaker
        self._last_metrics: Optional[dict] = None

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """
        Connect to the megakernel server, stream PCM frames, yield TTSAudioRawFrame.

        Called by Pipecat for each LLM text chunk that needs to be synthesized.
        """
        log.debug(f"MegakernelTTS synthesizing: [{text[:60]}]")

        try:
            ws_url = f"{self._server_url}/synthesize"
            async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
                # Send synthesis request
                await ws.send(json.dumps({"text": text, "speaker": self._speaker}))
                await self.start_tts_usage_metrics(text)

                first_chunk = True
                async for message in ws:
                    if isinstance(message, bytes):
                        # PCM audio chunk — push immediately to Pipecat
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False

                        yield TTSAudioRawFrame(
                            audio=message,
                            sample_rate=SAMPLE_RATE,
                            num_channels=NUM_CHANNELS,
                            context_id=context_id,
                        )

                    elif isinstance(message, str):
                        # JSON control message
                        data = json.loads(message)
                        if data.get("type") == "done":
                            self._last_metrics = data.get("metrics")
                            log.debug(f"TTS done: {self._last_metrics}")
                        elif data.get("type") == "error":
                            yield ErrorFrame(error=f"TTS server: {data.get('message')}")
                            return

        except ConnectionClosed as e:
            log.warning(f"TTS server connection closed: {e}")
            yield ErrorFrame(error=f"TTS server disconnected: {e}")
        except OSError as e:
            log.error(f"Cannot connect to TTS server at {self._server_url}: {e}")
            yield ErrorFrame(
                error=f"Cannot reach megakernel TTS server ({self._server_url}). "
                "Is `python -m server.app` running?"
            )
        except Exception as e:
            log.exception(f"TTS error: {e}")
            yield ErrorFrame(error=f"TTS synthesis failed: {e}")

    @property
    def last_metrics(self) -> Optional[dict]:
        """Metrics from the most recent synthesis (TTFC, RTF, tok/s)."""
        return self._last_metrics
