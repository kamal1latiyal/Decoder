"""
CodecDecoder — streaming wrapper over the official Qwen3-TTS-Tokenizer-12Hz
codec (model.speech_tokenizer on the loaded Qwen3TTSForConditionalGeneration).

Frame layout from config.json + qwen-tts source:
  frame_rate   : 12.5 Hz   (1 token = 80 ms of audio)
  sample_rate  : 24_000 Hz
  num_codebooks: 16        (1 semantic + 15 RVQ)
  output       : float32 wav in [-1, 1], converted to int16 LE PCM here

The official speech_tokenizer.decode() takes a list of dicts
[{ "audio_codes": Tensor[T, Q] }] and returns a list of full waveforms — it
is NOT designed for chunked streaming.  We work around this by carrying a
small overlap of previously-decoded frames and re-decoding the tail each call,
then emitting only the new samples.  This adds ~one chunk of latency but
preserves the decoder's intended causal context.

Future v2: hook the codec decoder's internal sliding-window state directly
to avoid the overlap re-decode.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)   # 1920

# DEFAULT chunk and overlap. These are now both runtime-configurable per
# CodecDecoder instance — see scripts/test_codec_streaming.py for the trade-off
# analysis on small-K configurations.
#
# - chunk_frames: how many NEW frames the talker has to produce before the
#   codec emits its first PCM chunk. Sets the TTFC floor (= chunk_frames / 12.5).
#   Smaller is better for latency but means more codec calls per second of audio.
# - overlap_frames: carried-over frames from the prior call, decoded again as
#   causal context for the codec's transposed-convolution stack. Larger overlap
#   keeps quality steady when chunk_frames is small.
DEFAULT_CHUNK_FRAMES = 4
DEFAULT_OVERLAP_FRAMES = 4

# Hard lower bound: 1 frame is the smallest input the codec accepts (verified
# in test_codec_min_chunk.py). Anything below this is an error.
MIN_CHUNK_FRAMES = 1

# Legacy alias still used by pipeline.py / scripts; equals current default.
MIN_CHUNK_SAMPLES = DEFAULT_CHUNK_FRAMES * SAMPLES_PER_FRAME  # 7680
OVERLAP_FRAMES = DEFAULT_OVERLAP_FRAMES


@dataclass
class _DecodeHandle:
    """Handle returned by CodecDecoder.submit(). Holds a reference to the
    in-flight GPU work + the new_frame_count needed to slice the trailing
    PCM window from the eventual wav. Pass to .collect() to sync and read."""
    wav_tensor: Optional[torch.Tensor]      # GPU wav tensor, in-flight
    event: Optional[object]                  # torch.cuda.Event recorded on codec_stream, or None on CPU
    new_frame_count: int
    # On CPU fallback path, .pcm holds the already-computed bytes directly.
    pcm: Optional[bytes] = None


class CodecDecoder:
    """Streaming wrapper over qwen3_tts.speech_tokenizer."""

    # Worst-case input length = chunk_frames + overlap_frames. We size the
    # pre-allocated buffer for an aggressive [chunk=8, overlap=24] worst case.
    _MAX_INPUT_FRAMES = 32

    def __init__(
        self,
        qwen3_tts_model: torch.nn.Module,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
    ):
        """
        Args:
            chunk_frames: how many NEW frames per decode call (TTFC floor =
                chunk_frames / 12.5 sec). Default 4 (320 ms floor).
            overlap_frames: previously-emitted frames replayed as causal
                context. Default 4. Set higher if you reduce chunk_frames
                aggressively to maintain audio quality.
        """
        if chunk_frames < MIN_CHUNK_FRAMES:
            raise ValueError(f"chunk_frames must be >= {MIN_CHUNK_FRAMES}, got {chunk_frames}")
        if overlap_frames < 0:
            raise ValueError(f"overlap_frames must be >= 0, got {overlap_frames}")
        if chunk_frames + overlap_frames > self._MAX_INPUT_FRAMES:
            raise ValueError(
                f"chunk_frames ({chunk_frames}) + overlap_frames ({overlap_frames}) "
                f"= {chunk_frames + overlap_frames} > _MAX_INPUT_FRAMES "
                f"({self._MAX_INPUT_FRAMES}); raise the buffer cap if needed"
            )

        self.device = next(qwen3_tts_model.parameters()).device
        self._tokenizer = getattr(qwen3_tts_model, "speech_tokenizer", None)
        if self._tokenizer is None:
            raise AttributeError(
                "Loaded model has no `speech_tokenizer` attribute. "
                "Did you load Qwen3TTSForConditionalGeneration?"
            )
        self._chunk_frames = chunk_frames
        self._overlap_frames = overlap_frames
        self._history_frames: list[list[int]] = []   # last overlap_frames we already emitted

        # Pre-allocate the codec input tensor — avoids per-call torch.tensor()
        # which is ~0.5-2 ms on GPU due to host→device copy + alloc overhead.
        self._codes_buf = torch.zeros(
            self._MAX_INPUT_FRAMES, 16, dtype=torch.long, device=self.device,
        )

        # Dedicated CUDA stream for codec ops so they can overlap with talker
        # work on the default stream. None on CPU — submit()/collect() degrade
        # to a synchronous in-line call there.
        if self.device.type == "cuda":
            self._codec_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(device=self.device)
        else:
            self._codec_stream = None

    def reset(self) -> None:
        self._history_frames = []

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def min_chunk_frames(self) -> int:
        """How many new frames the caller must hand us per decode call.
        Returns the instance's configured chunk_frames (not the hard floor)."""
        return self._chunk_frames

    @property
    def overlap_frames(self) -> int:
        return self._overlap_frames

    @torch.inference_mode()
    def submit(self, frames: list[list[int]]) -> _DecodeHandle:
        """
        Submit a codec decode to the codec CUDA stream and return a handle.
        Does NOT block until completion — caller is free to issue more talker
        work on the default stream while the codec runs in parallel.

        On CPU (no CUDA stream available), this collapses to a synchronous
        decode and returns a handle with `.pcm` pre-filled.

        Pair with .collect(handle) to get the PCM bytes (may block).
        """
        if len(frames) < self._chunk_frames:
            raise ValueError(
                f"Need at least {self._chunk_frames} frames per decode; got {len(frames)}"
            )

        all_frames = self._history_frames + frames
        new_frame_count = len(frames)
        n_total = len(all_frames)

        # Rotate history NOW (before submit) so the caller sees the updated
        # state immediately; the GPU work continues async on its own stream
        # and doesn't touch _history_frames.
        keep = min(self._overlap_frames, len(all_frames))
        self._history_frames = list(all_frames[-keep:])

        # Stage codec ids into pre-allocated GPU buffer.
        if n_total > self._MAX_INPUT_FRAMES:
            codes_2d = torch.tensor(all_frames, dtype=torch.long, device=self.device)
        else:
            self._codes_buf[:n_total].copy_(
                torch.as_tensor(all_frames, dtype=torch.long)
            )
            codes_2d = self._codes_buf[:n_total]

        # CPU path: just decode synchronously, return a handle with pcm baked in.
        if self._codec_stream is None:
            wavs, fs = self._tokenizer.decode([{"audio_codes": codes_2d}])
            assert fs == SAMPLE_RATE, f"Codec sample rate {fs} != expected {SAMPLE_RATE}"
            wav = wavs[0]
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().float().cpu().numpy()
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            pcm = _slice_and_pack(wav, new_frame_count)
            return _DecodeHandle(wav_tensor=None, event=None, new_frame_count=new_frame_count, pcm=pcm)

        # GPU path: run the codec on the codec stream. Make sure the codec
        # stream sees the codes_2d writes from the default stream (we just
        # copy_'d into _codes_buf on the default stream).
        self._codec_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._codec_stream):
            wavs, fs = self._tokenizer.decode([{"audio_codes": codes_2d}])
            assert fs == SAMPLE_RATE, f"Codec sample rate {fs} != expected {SAMPLE_RATE}"
            wav_tensor = wavs[0]
            # Don't .cpu() here — that would force a sync. Just hold the GPU
            # tensor and let .collect() do the host copy when the consumer
            # actually needs the bytes.

        event = torch.cuda.Event()
        event.record(self._codec_stream)

        return _DecodeHandle(
            wav_tensor=wav_tensor,
            event=event,
            new_frame_count=new_frame_count,
            pcm=None,
        )

    def collect(self, handle: _DecodeHandle) -> bytes:
        """
        Block until the submit's codec work is done, then return PCM bytes.
        Idempotent: calling collect() twice on the same handle returns the
        same bytes (no re-sync, no re-copy) thanks to the .pcm cache.
        """
        if handle.pcm is not None:
            return handle.pcm

        # Wait for the codec stream to finish this handle's work.
        if handle.event is not None:
            handle.event.synchronize()

        wav = handle.wav_tensor
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        pcm = _slice_and_pack(wav, handle.new_frame_count)
        handle.pcm = pcm  # cache for any future .collect() on this handle
        return pcm

    def decode(self, frames: list[list[int]]) -> bytes:
        """Synchronous decode for backwards compatibility / simple use.
        Equivalent to submit(frames) immediately followed by collect()."""
        return self.collect(self.submit(frames))


def _slice_and_pack(wav: np.ndarray, new_frame_count: int) -> bytes:
    """Take the codec's float32 wav, slice trailing `new_frame_count *
    SAMPLES_PER_FRAME` samples, clip + pack to int16 PCM."""
    new_samples = min(new_frame_count * SAMPLES_PER_FRAME, wav.shape[0])
    new_wav = wav[-new_samples:]
    clipped = np.clip(new_wav, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
