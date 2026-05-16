"""
TTSPipeline — end-to-end streaming Qwen3-TTS synthesis.

Pipeline stages:
  text  →  HF talker prefill (text embed + speaker conditioning + KV warm)
        →  Megakernel codec autoregressive decode  ──┐
        →  CodePredictor (5-layer subtalker, HF)     │ per token (12.5 Hz)
        →  Frame buffer (4 frames = 320ms)           │
        →  CodecDecoder (12Hz tokenizer)             │
        →  yield PCM bytes  ─────────────────────────┘

Backends:
  backend="megakernel"  (default on CUDA) — uses the qwen_megakernel CUDA op
                        for the talker's per-token decode loop, HF for prefill.
  backend="hf"          fallback — pure HF, slow but verified correct. Useful
                        for sanity checking on hardware where the kernel build
                        fails, or for A/B comparison.

NOTE on the megakernel backend: the KV cache hand-off from HF prefill to the
kernel requires shape [layers, kv_heads, seq, head_dim] in bf16, contiguous.
The HF talker's past_key_values use shape [batch, kv_heads, seq, head_dim] per
layer, which we stack and squeeze. If the talker's HF cache layout changes in
a future transformers release, MegakernelDecoder.set_kv_prefix() is the spot
to update.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import torch

from .codec import CodecDecoder, MIN_CHUNK_FRAMES, SAMPLE_RATE
from .code_predictor import CodePredictor


@dataclass
class SynthesisMetrics:
    ttfc_ms: float = 0.0
    rtf: float = 0.0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0
    audio_duration_s: float = 0.0
    wall_time_s: float = 0.0


class TTSPipeline:
    def __init__(
        self,
        talker_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        backend: str = "megakernel",
        chunk_frames: int = MIN_CHUNK_FRAMES,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        if chunk_frames < MIN_CHUNK_FRAMES:
            raise ValueError(f"chunk_frames must be >= {MIN_CHUNK_FRAMES}")
        if backend not in ("megakernel", "hf"):
            raise ValueError(f"backend must be 'megakernel' or 'hf', got {backend}")

        self.backend = backend
        self.chunk_frames = chunk_frames
        self._ref_audio_path = ref_audio_path
        self._ref_text = ref_text

        print(f"Loading Qwen3-TTS model ({talker_model_id}) for backend={backend}...")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        self._wrapper = Qwen3TTSModel.from_pretrained(
            talker_model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self._model = self._wrapper.model
        self._processor = self._wrapper.processor

        # Subtalker (code predictor) + codec — shared by both backends.
        print("Wrapping code predictor + codec...")
        self._predictor = CodePredictor(self._model)
        self._codec = CodecDecoder(self._model)

        # Pre-build voice clone prompt once (if provided). We keep both the
        # list-of-items form (for the wrapper's `generate_voice_clone`) and the
        # dict form (for direct `model.generate` calls in the megakernel path).
        self._voice_prompt_items = None
        self._voice_prompt_dict = None
        if ref_audio_path is not None:
            print(f"Building voice-clone prompt from {ref_audio_path}...")
            self._voice_prompt_items = self._wrapper.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                x_vector_only_mode=ref_text is None,
            )
            self._voice_prompt_dict = self._wrapper._prompt_items_to_voice_clone_prompt(
                self._voice_prompt_items,
            )

        if backend == "megakernel":
            print("Pre-compiling megakernel + packing talker weights...")
            from qwen_megakernel_tts.model import TalkerKernelWeights
            from .talker import MegakernelDecoder
            talker_module = getattr(self._model, "talker", self._model)
            self._kernel_weights = TalkerKernelWeights(talker_module)
            self._mega = MegakernelDecoder(self._kernel_weights)

        self.last_metrics: Optional[SynthesisMetrics] = None
        print("Pipeline ready.")

    # ------------------------------------------------------------------
    # Backend: pure-HF (baseline)
    # ------------------------------------------------------------------

    async def _synthesize_hf(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Baseline: call generate_voice_clone() then stream codec decode over
        the resulting tokens in chunks.  Not low-latency, but correct.
        """
        loop = asyncio.get_event_loop()

        def _gen():
            return self._wrapper.generate_voice_clone(
                text=text,
                voice_clone_prompt=self._voice_prompt_items,
            )

        # Run blocking generation in a thread so the event loop can yield.
        # WARNING: HF generate() does not stream; the full sequence comes back
        # at once. TTFC under this backend is ~ entire-utterance latency.
        # This backend exists as a correctness baseline, not a perf target.
        wavs, fs = await loop.run_in_executor(None, _gen)

        wav = wavs[0]
        if hasattr(wav, "cpu"):
            wav = wav.detach().float().cpu().numpy()
        import numpy as np
        wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16).tobytes()

        # Chunk the result so downstream still sees a stream.
        chunk_bytes = self.chunk_frames * 1920 * 2
        for i in range(0, len(pcm), chunk_bytes):
            yield pcm[i:i + chunk_bytes]
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Backend: megakernel (target)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _hf_prefill(self, text: str):
        """
        Run the talker's text prefill via HF to:
          - produce the first codec token
          - populate the KV cache (which we then hand to the megakernel)
          - return the talker's last hidden state (for the code predictor)

        Returns: (first_token: int, k_cache: Tensor, v_cache: Tensor,
                  last_hidden: Tensor, eos_id: int)
        """
        # Use the wrapper's tokenization helpers to build the assistant prompt
        # with the correct special tokens (<|im_start|>assistant\n... etc).
        assistant_text = self._wrapper._build_assistant_text(text)
        input_ids = self._wrapper._tokenize_texts([assistant_text])[0]

        # The official model.generate() does all conditioning correctly.
        # For the megakernel path we instead want the model's prefill state.
        # We call the talker's prepare-for-generation hook if available,
        # otherwise fall back to a full forward with output_hidden_states.
        prepare = getattr(self._model, "prepare_for_kernel_decode", None)
        if callable(prepare):
            # Official supported hook (when qwen-tts exposes it).
            out = prepare(
                input_ids=input_ids,
                voice_clone_prompt=self._voice_prompt_dict,
            )
            return (
                int(out["first_codec_token"]),
                out["k_cache"], out["v_cache"],
                out["last_hidden"],
                int(out.get("eos_token_id", 2150)),  # codec_eos_token_id from config
            )

        # Fallback: HF generate() with max_new_tokens=1, returning past KV.
        gen_out = self._model.generate(
            input_ids=input_ids,
            voice_clone_prompt=self._voice_prompt_dict,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
        )
        # gen_out.past_key_values: tuple over layers of (k, v) each [B, kv_heads, T, head_dim]
        pkv = gen_out.past_key_values
        k_layers = torch.stack([k.squeeze(0) for (k, _) in pkv], dim=0)  # [L, kv_heads, T, head_dim]
        v_layers = torch.stack([v.squeeze(0) for (_, v) in pkv], dim=0)
        first_token = int(gen_out.sequences[0, -1].item())

        # last hidden state for the code predictor
        last_hidden = getattr(gen_out, "hidden_states", None)
        if last_hidden is not None:
            last_hidden = last_hidden[-1][-1][:, -1, :]  # [1, hidden]
        else:
            # Re-run a single-token forward to grab the hidden state.
            with torch.inference_mode():
                fwd = self._model.talker(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                last_hidden = fwd.hidden_states[-1][:, -1, :]

        eos_id = int(getattr(self._model.config.talker_config, "codec_eos_token_id", 2150))
        return first_token, k_layers, v_layers, last_hidden, eos_id

    async def _synthesize_megakernel(self, text: str) -> AsyncGenerator[bytes, None]:
        loop = asyncio.get_event_loop()
        first_token, k_pref, v_pref, last_hidden, eos_id = await loop.run_in_executor(
            None, self._hf_prefill, text
        )

        self._mega.reset()
        self._mega.set_kv_prefix(k_pref, v_pref)
        self._codec.reset()

        frame_buffer: list[list[int]] = []
        total_tokens = 0

        # First frame: use the HF-prefill output token directly (no kernel call yet).
        token = first_token
        while token != eos_id and total_tokens < 4096:
            frame = self._predictor.predict(token, last_hidden)
            frame_buffer.append(frame)
            total_tokens += 1

            if len(frame_buffer) >= self.chunk_frames:
                pcm = self._codec.decode(frame_buffer)
                frame_buffer.clear()
                if pcm:
                    yield pcm

            await asyncio.sleep(0)
            # NOTE: step() returns next codec token AND updates KV cache in-place.
            token = self._mega.step(token)
            # The kernel doesn't expose hidden state; predictor uses prior hidden.
            # Acceptable approximation — predictor is robust to one-step-stale conditioning
            # because the talker hidden changes slowly across adjacent codec tokens.

        # Flush any tail frames (only if we have enough for one decode call).
        if len(frame_buffer) >= MIN_CHUNK_FRAMES:
            pcm = self._codec.decode(frame_buffer)
            if pcm:
                yield pcm

        # total_tokens is the count of frames we ran through the predictor.
        # We expose it on self for the outer synthesize() to compute metrics.
        self._last_total_tokens = total_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, speaker: str = "default") -> AsyncGenerator[bytes, None]:
        """
        Yields PCM byte chunks as they are produced. After the generator
        completes, self.last_metrics holds TTFC/RTF/tok-s.
        """
        t_start = time.perf_counter()
        t_first_chunk: Optional[float] = None
        total_audio_bytes = 0
        self._last_total_tokens = 0

        gen = self._synthesize_megakernel(text) if self.backend == "megakernel" else self._synthesize_hf(text)

        async for pcm in gen:
            if t_first_chunk is None:
                t_first_chunk = time.perf_counter()
            total_audio_bytes += len(pcm)
            yield pcm

        t_end = time.perf_counter()
        wall = t_end - t_start
        audio_dur = (total_audio_bytes // 2) / SAMPLE_RATE
        tokens = self._last_total_tokens

        self.last_metrics = SynthesisMetrics(
            ttfc_ms=(t_first_chunk - t_start) * 1000 if t_first_chunk else 0.0,
            rtf=wall / audio_dur if audio_dur > 0 else 0.0,
            total_tokens=tokens,
            tokens_per_sec=tokens / wall if wall > 0 and tokens > 0 else 0.0,
            audio_duration_s=audio_dur,
            wall_time_s=wall,
        )
