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

import numpy as np
import torch
from typing import Optional


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
    def decode(self, frames: list[list[int]]) -> bytes:
        """
        Decode the newest `frames` (each a list of 16 codebook ids) into PCM.

        Streaming strategy: decode `history + frames` together so the codec
        has its causal context, then emit ONLY the samples corresponding to
        the new `frames` (the trailing block of the decoded wav).

        PREVIOUS BUG (now fixed): the old version tracked `_emitted_samples`
        as `wav.shape[0]` (the cumulative length of the *current* decode).
        After the first chunk, every subsequent decode of `[history(4) +
        new(4)]` produces the same total length (8 frames worth ≈ 15,360
        samples), so `skip_samples >= wav.shape[0]` and zero new samples
        were emitted. Audio truncated silently after one chunk.
        Fix: slice the trailing `len(frames) * SAMPLES_PER_FRAME` samples.
        """
        if len(frames) < self._chunk_frames:
            raise ValueError(
                f"Need at least {self._chunk_frames} frames per decode; got {len(frames)}"
            )

        all_frames = self._history_frames + frames
        new_frame_count = len(frames)
        n_total = len(all_frames)
        if n_total > self._MAX_INPUT_FRAMES:
            # Should never happen in practice (overlap=4 + chunk=4 = 8), but
            # be defensive: fall back to per-call allocation if someone calls
            # with a huge chunk.
            codes_2d = torch.tensor(all_frames, dtype=torch.long, device=self.device)
        else:
            # Reuse the pre-allocated buffer: copy ids in (host → device) then
            # slice to the actual length. `as_tensor` avoids one extra copy
            # relative to `torch.tensor` when the source is already a list.
            self._codes_buf[:n_total].copy_(
                torch.as_tensor(all_frames, dtype=torch.long)
            )
            codes_2d = self._codes_buf[:n_total]
        wavs, fs = self._tokenizer.decode([{"audio_codes": codes_2d}])
        assert fs == SAMPLE_RATE, f"Codec sample rate {fs} != expected {SAMPLE_RATE}"
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)

        # Emit only the trailing window corresponding to the *new* frames.
        # This is intentionally frame-count-based, not absolute-sample-based:
        # the codec output for `history + new` is a fresh decode each call,
        # not a continuation of the prior wav, so tracking cumulative emitted
        # samples doesn't make sense.
        new_samples = min(new_frame_count * SAMPLES_PER_FRAME, wav.shape[0])
        new_wav = wav[-new_samples:]

        # Rotate history: keep last overlap_frames for next call's causal context.
        keep = min(self._overlap_frames, len(all_frames))
        self._history_frames = list(all_frames[-keep:])

        clipped = np.clip(new_wav, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()
