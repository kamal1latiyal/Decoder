"""
TTSPipeline — end-to-end streaming Qwen3-TTS synthesis.

Pipeline stages:
  text  ─▶ HF talker prefill (builds inputs_embeds, runs one talker.model.forward
                              to warm KV cache, applies codec_head → first group-0 token)
        ─▶ Megakernel codec decode loop
              per step:
                subtalker(group-0, past_hidden) → groups 1..15
                next_input = Σᵢ embedᵢ(groupᵢ) + trailing_text_hidden[step]
                kernel.step_from_hidden(next_input) → next group-0 + post-norm hidden
        ─▶ CodecDecoder (12 Hz tokenizer) → PCM @ 24 kHz mono int16

Backends:
  backend="megakernel" (default on CUDA) — uses the qwen_megakernel CUDA op
                       for the talker's per-token decode loop, HF for prefill + subtalker.
  backend="hf"         baseline — pure HF wrapper.generate_voice_clone(), full sync.
                       Verified correct; not low-latency (whole utterance latency = TTFC).

KV CACHE HAND-OFF
-----------------
HF prefill returns past_key_values as a DynamicCache. Each `cache.layers[i]`
exposes `.keys` and `.values` of shape [B=1, kv_heads=8, T, head_dim=128].
We stack across layers and squeeze the batch dim to produce
[layers=28, kv_heads=8, T, head_dim=128] which MegakernelDecoder.set_kv_prefix
expects. Verified hardware-side TODO.

MONKEY-PATCH PREFILL
--------------------
The wrapper (Qwen3TTSModel.generate_voice_clone) constructs talker_input_embeds
from text + codec prefill tokens + speaker embed + special tokens. Re-implementing
that ourselves would duplicate ~80 lines of fragile logic. Instead we replace
`model.talker.generate` with a one-shot forward, run the wrapper, and intercept
the result via a sentinel exception. The wrapper builds everything correctly;
our patched `generate` just stops after prefill and exfiltrates the state.
"""

import asyncio
import time
import types
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import torch

from .codec import CodecDecoder, MIN_CHUNK_FRAMES, SAMPLE_RATE
from .code_predictor import CodePredictor, CUDAGraphedCodePredictor


@dataclass
class SynthesisMetrics:
    ttfc_ms: float = 0.0
    rtf: float = 0.0
    total_tokens: int = 0          # number of decoded frames (one per codec step)
    tokens_per_sec: float = 0.0    # talker steps / second (megakernel throughput)
    audio_duration_s: float = 0.0
    wall_time_s: float = 0.0


class _PrefillDone(Exception):
    """Sentinel raised by the monkey-patched talker.generate after one forward pass."""
    def __init__(self, payload: dict):
        self.payload = payload


def _extract_kv_for_kernel(cache) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a transformers DynamicCache into the layout MegakernelDecoder expects.

    Returns (k, v) each of shape [num_layers, num_kv_heads, prefix_len, head_dim],
    dtype bf16, contiguous, on cuda.
    """
    if not hasattr(cache, "layers"):
        raise TypeError(
            f"Expected a DynamicCache (transformers>=4.50); got {type(cache).__name__}. "
            "If this is a legacy tuple-of-tuples, convert via DynamicCache.from_legacy_cache(...)."
        )
    k_list, v_list = [], []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            raise RuntimeError("DynamicCache layer is uninitialised — prefill produced no KV.")
        # layer.keys / .values shape: [B=1, kv_heads, T, head_dim]
        k_list.append(layer.keys.squeeze(0))
        v_list.append(layer.values.squeeze(0))
    k = torch.stack(k_list, dim=0).contiguous().to(torch.bfloat16)
    v = torch.stack(v_list, dim=0).contiguous().to(torch.bfloat16)
    return k, v


class TTSPipeline:
    def __init__(
        self,
        talker_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        backend: str = "megakernel",
        chunk_frames: int = MIN_CHUNK_FRAMES,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        use_cuda_graph: bool = True,
    ):
        """
        Args:
            use_cuda_graph: when True (default) and backend == 'megakernel',
                use the CUDAGraphedCodePredictor — captures the subtalker's
                15-step decode into a single CUDA graph, ~8x faster per
                frame than HF's GenerationMixin path. Set False to A/B
                against the HF reference. Greedy-only (no sampling)
                because graph capture needs deterministic ops.
        """
        if chunk_frames < MIN_CHUNK_FRAMES:
            raise ValueError(f"chunk_frames must be >= {MIN_CHUNK_FRAMES}")
        if backend not in ("megakernel", "hf"):
            raise ValueError(f"backend must be 'megakernel' or 'hf', got {backend}")

        self.backend = backend
        self.chunk_frames = chunk_frames
        self._ref_audio_path = ref_audio_path
        self._ref_text = ref_text
        self._language = language
        self._use_cuda_graph = use_cuda_graph and backend == "megakernel"

        print(f"Loading Qwen3-TTS model ({talker_model_id}) for backend={backend}...")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        self._wrapper = Qwen3TTSModel.from_pretrained(
            talker_model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self._model = self._wrapper.model            # Qwen3TTSForConditionalGeneration
        self._processor = self._wrapper.processor

        # Subtalker (code predictor) + codec — shared by both backends.
        # CUDA-graphed predictor is the optimised path; falls back to HF
        # CodePredictor on capture failure or when use_cuda_graph=False.
        if self._use_cuda_graph:
            print("Wrapping code predictor (CUDA-graphed, greedy) + codec...")
            try:
                self._predictor = CUDAGraphedCodePredictor(self._model)
                print("  ✓ CUDA graph captured for subtalker")
            except Exception as e:
                print(f"  ⚠ CUDA graph capture failed: {e!r}")
                print("  ↳ falling back to HF CodePredictor (slower)")
                self._predictor = CodePredictor(self._model)
                self._use_cuda_graph = False
        else:
            print("Wrapping code predictor (HF reference) + codec...")
            self._predictor = CodePredictor(self._model)
        self._codec = CodecDecoder(self._model)

        # Pre-build voice clone prompt once (if provided).  The wrapper handles
        # both ICL (ref_text given) and x-vector-only (ref_text=None) modes.
        self._voice_prompt_items = None
        if ref_audio_path is not None:
            print(f"Building voice-clone prompt from {ref_audio_path}...")
            self._voice_prompt_items = self._wrapper.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                x_vector_only_mode=ref_text is None,
            )

        # Cached configuration values.
        self._eos_token_id = int(self._model.config.talker_config.codec_eos_token_id)
        # Frame cap. 12.5 Hz frame rate, so 128 frames ≈ 10s of audio per
        # request — plenty for a conversational voice agent. Lower than the
        # upstream cap (2048 ≈ 164s) because the megakernel uses greedy decoding
        # which sometimes fails to emit eos. With sampling-aware decoding this
        # could safely go higher; greedy works best with a tight bound.
        self._max_new_tokens = 128

        if backend == "megakernel":
            print("Pre-compiling megakernel + packing talker weights...")
            from qwen_megakernel_tts.model import TalkerKernelWeights
            from .talker import MegakernelDecoder
            self._kernel_weights = TalkerKernelWeights(self._model.talker)
            self._mega = MegakernelDecoder(self._kernel_weights)

        self.last_metrics: Optional[SynthesisMetrics] = None
        print("Pipeline ready.")

    # ------------------------------------------------------------------
    # Backend: pure-HF (baseline)
    # ------------------------------------------------------------------

    async def _synthesize_hf(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Baseline: call generate_voice_clone() then stream codec decode over
        the resulting tokens in chunks.  Not low-latency (HF blocks until full
        sequence is produced), but correctness-verified by qwen_tts maintainers.
        """
        loop = asyncio.get_event_loop()

        def _gen():
            kwargs = dict(text=text, language=self._language)
            if self._voice_prompt_items is not None:
                kwargs["voice_clone_prompt"] = self._voice_prompt_items
            else:
                # No ref audio at all — fall back to x-vector default by passing
                # ref_audio=None and x_vector_only_mode=True via a noop prompt.
                # Easiest is to require a ref or use the wrapper's fallback path:
                # for now, raise so the user sees the explicit requirement.
                raise RuntimeError(
                    "HF backend requires a reference voice. Pass ref_audio_path "
                    "to TTSPipeline(...) or use backend='megakernel' with the "
                    "default speaker embedding path (which the wrapper supports)."
                )
            return self._wrapper.generate_voice_clone(**kwargs)

        wavs, fs = await loop.run_in_executor(None, _gen)

        import numpy as np
        wav = wavs[0]
        if hasattr(wav, "cpu"):
            wav = wav.detach().float().cpu().numpy()
        wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16).tobytes()

        # Chunk the result so downstream still sees a stream.
        chunk_bytes = self.chunk_frames * 1920 * 2
        for i in range(0, len(pcm), chunk_bytes):
            yield pcm[i:i + chunk_bytes]
            await asyncio.sleep(0)

        # HF backend can't break out tokens-per-sec cleanly, so leave that as 0.
        self._last_total_tokens = 0

    # ------------------------------------------------------------------
    # Backend: megakernel (target)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _hf_prefill(self, text: str) -> dict:
        """
        Run the wrapper's text/speaker/codec prefix construction and the talker
        prefill forward, intercepted via a monkey-patched talker.generate.

        Returns dict:
          first_token          : int                       (talker's first group-0 emission)
          kv_cache             : DynamicCache              (warm talker KV)
          last_hidden          : Tensor [1, 1, 1024]       (post-norm hidden at end of prefill)
          trailing_text_hidden : Tensor [1, K, 1024]       (remaining text for streaming)
          tts_pad_embed        : Tensor [1, 1, 1024]       (pad embedding for past-trailing steps)
        """
        if self._voice_prompt_items is None:
            # Build a default x-vector prompt the first time we synthesize without ref audio.
            # qwen_tts's wrapper requires either ref_audio or voice_clone_prompt; the simplest
            # default path is to require the caller pass ref_audio_path to TTSPipeline.
            # For full prompt-less synthesis we'd need a custom speaker embedding seed.
            raise RuntimeError(
                "Megakernel backend currently requires a reference voice "
                "(ref_audio_path argument to TTSPipeline). Default-voice synthesis "
                "needs a default x-vector seed that qwen_tts doesn't expose directly."
            )

        captured: dict = {}

        def patched_generate(self_tl, inputs_embeds=None, attention_mask=None,
                             trailing_text_hidden=None, tts_pad_embed=None, **kw):
            """
            Replacement for Qwen3TTSTalkerForConditionalGeneration.generate.
            Runs ONE forward through the talker's base transformer, captures
            (kv, last_hidden, first_token), and raises _PrefillDone to break out
            of the wrapper.
            """
            out = self_tl.model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )
            # last_hidden_state already has the final norm applied
            # (Qwen3TTSTalkerModel.forward applies self.norm before returning).
            last_hidden = out.last_hidden_state[:, -1:, :]            # [1, 1, 1024]
            logits = self_tl.codec_head(last_hidden)                   # [1, 1, 3072]
            first_token = int(logits.argmax(dim=-1).item())

            payload = {
                "first_token": first_token,
                "kv_cache": out.past_key_values,
                "last_hidden": last_hidden,
                "trailing_text_hidden": trailing_text_hidden,
                "tts_pad_embed": tts_pad_embed,
            }
            raise _PrefillDone(payload)

        original = self._model.talker.generate
        self._model.talker.generate = types.MethodType(patched_generate, self._model.talker)
        try:
            try:
                self._wrapper.generate_voice_clone(
                    text=text,
                    language=self._language,
                    voice_clone_prompt=self._voice_prompt_items,
                    do_sample=False,                    # greedy for determinism
                    max_new_tokens=1,                   # honoured nowhere because we raise first
                )
            except _PrefillDone as done:
                captured = done.payload
            else:
                raise RuntimeError(
                    "Monkey-patched prefill did not fire — wrapper completed without "
                    "calling talker.generate. Did qwen_tts change its generate path?"
                )
        finally:
            self._model.talker.generate = original

        return captured

    async def _synthesize_megakernel(self, text: str) -> AsyncGenerator[bytes, None]:
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(None, self._hf_prefill, text)

        first_token: int = payload["first_token"]
        kv_cache = payload["kv_cache"]
        last_hidden: torch.Tensor = payload["last_hidden"]               # [1, 1, 1024]
        trailing_text_hidden: torch.Tensor = payload["trailing_text_hidden"]  # [1, K, 1024]
        tts_pad_embed: torch.Tensor = payload["tts_pad_embed"]           # [1, 1, 1024]

        # Push KV prefix into the kernel.
        k_pref, v_pref = _extract_kv_for_kernel(kv_cache)
        self._mega.reset()
        self._mega.set_kv_prefix(k_pref, v_pref)
        self._codec.reset()

        eos_id = self._eos_token_id
        trailing_len = trailing_text_hidden.shape[1]
        device = last_hidden.device
        dtype = last_hidden.dtype

        token = first_token
        past_hidden = last_hidden                # [1, 1, 1024]
        step = 0
        frame_buffer: list[list[int]] = []
        total_tokens = 0

        while token != eos_id and total_tokens < self._max_new_tokens:
            # Run subtalker → full 16-group frame + summed codec embedding.
            frame, codec_hidden_sum = await loop.run_in_executor(
                None,
                lambda t=token, h=past_hidden: self._predictor.predict(t, h),
            )
            frame_buffer.append(frame)
            total_tokens += 1

            if len(frame_buffer) >= self.chunk_frames:
                pcm = await loop.run_in_executor(
                    None, self._codec.decode, list(frame_buffer)
                )
                frame_buffer.clear()
                if pcm:
                    yield pcm

            # Build next-step input embedding (matches talker.forward, lines 1687–1692).
            if step < trailing_len:
                next_input = codec_hidden_sum + trailing_text_hidden[:, step : step + 1]
            else:
                next_input = codec_hidden_sum + tts_pad_embed
            next_input = next_input.to(device=device, dtype=dtype).reshape(-1)  # [1024]

            # Kernel step → next group-0 token + post-norm hidden.
            token, post_norm = self._mega.step_from_hidden(next_input)
            past_hidden = post_norm.view(1, 1, -1)
            step += 1

            await asyncio.sleep(0)  # yield to event loop so PCM dispatch interleaves

        # Flush any trailing frames (only if we have enough for one codec decode).
        if len(frame_buffer) >= MIN_CHUNK_FRAMES:
            pcm = await loop.run_in_executor(
                None, self._codec.decode, list(frame_buffer)
            )
            if pcm:
                yield pcm

        self._last_total_tokens = total_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, speaker: str = "default") -> AsyncGenerator[bytes, None]:
        """
        Yields PCM byte chunks as they are produced. After the generator
        completes, `self.last_metrics` holds TTFC/RTF/tok-per-s.
        """
        t_start = time.perf_counter()
        t_first_chunk: Optional[float] = None
        total_audio_bytes = 0
        self._last_total_tokens = 0

        gen = (
            self._synthesize_megakernel(text)
            if self.backend == "megakernel"
            else self._synthesize_hf(text)
        )

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
