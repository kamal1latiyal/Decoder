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
MIN_CHUNK_FRAMES = 4                                # codec's minimum decode window
MIN_CHUNK_SAMPLES = MIN_CHUNK_FRAMES * SAMPLES_PER_FRAME  # 7680
OVERLAP_FRAMES = 4                                  # frames carried across calls


class CodecDecoder:
    """Streaming wrapper over qwen3_tts.speech_tokenizer."""

    def __init__(self, qwen3_tts_model: torch.nn.Module):
        self.device = next(qwen3_tts_model.parameters()).device
        self._tokenizer = getattr(qwen3_tts_model, "speech_tokenizer", None)
        if self._tokenizer is None:
            raise AttributeError(
                "Loaded model has no `speech_tokenizer` attribute. "
                "Did you load Qwen3TTSForConditionalGeneration?"
            )
        self._history_frames: list[list[int]] = []   # last OVERLAP_FRAMES we already emitted
        self._emitted_samples: int = 0               # cumulative samples already pushed out

    def reset(self) -> None:
        self._history_frames = []
        self._emitted_samples = 0

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def min_chunk_frames(self) -> int:
        return MIN_CHUNK_FRAMES

    @torch.inference_mode()
    def decode(self, frames: list[list[int]]) -> bytes:
        """
        Decode the newest `frames` (each a list of 16 codebook ids) into PCM.
        Returns only the audio samples that haven't been emitted before.

        Sample-accurate truncation against `_emitted_samples` makes the stream
        gap-free even though we re-decode the overlap each call.
        """
        if len(frames) < MIN_CHUNK_FRAMES:
            raise ValueError(
                f"Need at least {MIN_CHUNK_FRAMES} frames per decode; got {len(frames)}"
            )

        # Concatenate overlap (already-emitted tail) + new frames, decode together.
        all_frames = self._history_frames + frames
        codes_2d = torch.tensor(all_frames, dtype=torch.long, device=self.device)  # [T, 16]
        # speech_tokenizer.decode() expects per-sample dicts of {"audio_codes": [T, Q]}
        wavs, fs = self._tokenizer.decode([{"audio_codes": codes_2d}])
        assert fs == SAMPLE_RATE, f"Codec sample rate {fs} != expected {SAMPLE_RATE}"
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)

        # Drop the leading samples corresponding to already-emitted frames.
        # Use sample-accurate count rather than frames*1920 to avoid drift on
        # codecs that produce slightly fewer samples at sequence start.
        skip_samples = self._emitted_samples
        new_wav = wav[skip_samples:] if skip_samples < wav.shape[0] else wav[:0]

        self._emitted_samples = wav.shape[0]

        # Rotate history: keep the last OVERLAP_FRAMES of frames for next call's context.
        keep = OVERLAP_FRAMES if len(all_frames) > OVERLAP_FRAMES else len(all_frames)
        self._history_frames = list(all_frames[-keep:])

        clipped = np.clip(new_wav, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()
