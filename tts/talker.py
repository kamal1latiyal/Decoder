"""
MegakernelDecoder — wraps the qwen_megakernel CUDA op for a single codec
autoregressive decode step.

This deliberately does NOT try to be a full Qwen3-TTS talker. It only owns:
  - the kernel call (torch.ops.qwen_megakernel_C.decode)
  - the KV cache that the kernel reads/writes
  - the RoPE tables (theta=1_000_000 for Qwen3-TTS)
  - the scratch buffers the kernel needs

It is fed prompt context (text + speaker conditioning) by writing into the KV
cache externally — see tts/pipeline.py for how the prefill is performed via
the official HF talker forward, then handed off to this decoder for the
streaming codec loop.

The kernel uses standard RoPE; the talker config actually specifies MRoPE
with mrope_section=[24,20,20] (separate scales for temporal / height / width
position dims).  We treat all position dims as the same — acceptable for
single-utterance speech since there is no image/video axis.  This is the
documented approximation, see DESIGN.md §4.
"""

import asyncio
import math
from typing import AsyncGenerator, Optional

import torch

from qwen_megakernel_tts.build import get_extension
from qwen_megakernel_tts.model import TalkerKernelWeights


# Talker architecture constants — matched against config.json of
# Qwen/Qwen3-TTS-12Hz-0.6B-Base on 2026-05-16.
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = NUM_Q_HEADS * HEAD_DIM      # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM    # 1024
ROPE_THETA = 1_000_000               # Qwen3-TTS talker rope_theta
MAX_SEQ_LEN = 4096                   # cap for codec sequence
SCRATCH_BLOCKS = 4096                # matches upstream Decoder; must hold worst-case LM head blocks


class MegakernelDecoder:
    """
    Stateful single-token decoder, calling
    `torch.ops.qwen_megakernel_C.decode(...)` for each codec token.
    """

    def __init__(self, talker_kernel_weights: TalkerKernelWeights):
        """
        Args:
            talker_kernel_weights: already-extracted talker weights packed for
                the kernel.  See qwen_megakernel_tts.TalkerKernelWeights.
        """
        # JIT-compile (or load cached) the kernel, then bind the registered op.
        get_extension()
        self._decode = torch.ops.qwen_megakernel_C.decode

        self.device = torch.device("cuda")
        self._w = talker_kernel_weights
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position = 0

        self._allocate_buffers()
        self._build_rope_tables()

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------

    def _allocate_buffers(self) -> None:
        bf16 = dict(dtype=torch.bfloat16, device=self.device)
        f32 = dict(dtype=torch.float32, device=self.device)

        self.k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM, **bf16)
        self.v_cache = torch.zeros_like(self.k_cache)

        # Scratch (per upstream Decoder, exact dtypes + sizes).
        self.hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self.act = torch.empty(HIDDEN_SIZE, **f32)
        self.res = torch.empty(HIDDEN_SIZE, **f32)
        self.q = torch.empty(Q_SIZE, **f32)
        self.k = torch.empty(KV_SIZE, **f32)
        self.v = torch.empty(KV_SIZE, **f32)
        self.attn_out = torch.empty(Q_SIZE, **f32)
        self.mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self.norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self.bmax_vals = torch.empty(SCRATCH_BLOCKS, **f32)
        self.bmax_idxs = torch.empty(SCRATCH_BLOCKS, dtype=torch.int32, device=self.device)
        self.out_token = torch.empty(1, dtype=torch.int32, device=self.device)

    def _build_rope_tables(self) -> None:
        """RoPE cos/sin with theta=1_000_000 (Qwen3-TTS talker)."""
        positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        freqs = torch.outer(positions, inv_freq)            # [MAX, HEAD/2]
        # Upstream uses `.repeat(1, 2)` (concat-like for split-half RoPE).
        self.cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).to(self.device).contiguous()
        self.sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).to(self.device).contiguous()

    # ------------------------------------------------------------------
    # state mgmt
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._position = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    def set_kv_prefix(
        self,
        k_prefix: torch.Tensor,
        v_prefix: torch.Tensor,
    ) -> None:
        """
        Inject a precomputed KV-cache prefix produced by the HF talker prefill.

        Shapes (must match):
          k_prefix, v_prefix: [NUM_LAYERS, NUM_KV_HEADS, prefix_len, HEAD_DIM]
          dtype: bfloat16

        After this call, self.step(first_codec_token) will continue decoding
        from position = prefix_len.
        """
        if k_prefix.dim() != 4:
            raise ValueError(f"k_prefix must be 4D; got {k_prefix.shape}")
        if k_prefix.shape != v_prefix.shape:
            raise ValueError("k_prefix and v_prefix shapes differ")
        if k_prefix.shape[0] != NUM_LAYERS or k_prefix.shape[1] != NUM_KV_HEADS or k_prefix.shape[3] != HEAD_DIM:
            raise ValueError(
                f"k_prefix shape {tuple(k_prefix.shape)} must be "
                f"[{NUM_LAYERS}, {NUM_KV_HEADS}, prefix_len, {HEAD_DIM}]"
            )
        prefix_len = k_prefix.shape[2]
        if prefix_len > MAX_SEQ_LEN:
            raise ValueError(f"KV prefix len {prefix_len} > MAX_SEQ_LEN {MAX_SEQ_LEN}")

        # NB: this is the documented integration point with HF prefill.
        # The actual HF cache layout has to be transposed/copied to match
        # the kernel's [layer, head, seq, dim] order. Verified on hardware.
        self.k_cache[:, :, :prefix_len, :].copy_(k_prefix.to(self.device, torch.bfloat16))
        self.v_cache[:, :, :prefix_len, :].copy_(v_prefix.to(self.device, torch.bfloat16))
        self._position = prefix_len

    # ------------------------------------------------------------------
    # decode primitives
    # ------------------------------------------------------------------

    def step(self, token_id: int) -> int:
        """Single decode step. Returns the next codec token id (greedy argmax)."""
        if self._position >= MAX_SEQ_LEN:
            raise RuntimeError(f"Sequence exceeded MAX_SEQ_LEN={MAX_SEQ_LEN}")

        self._decode(
            self.out_token,
            int(token_id),
            self._w.embed_weight,
            self._w.layer_weights_packed,
            self._w.final_norm_weight,
            self._w.lm_head_weight,
            self.cos_table,
            self.sin_table,
            self.k_cache,
            self.v_cache,
            self.hidden,
            self.act,
            self.res,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.mlp_inter,
            self.norm_out,
            self.bmax_vals,
            self.bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return int(self.out_token.item())   # GPU→CPU sync; ~1us, required for streaming

    async def stream(
        self,
        first_token: int,
        eos_token_id: int,
        max_new_tokens: int = 2048,
    ) -> AsyncGenerator[int, None]:
        """
        Async generator: yields codec token ids one per call.
        Caller is responsible for prefilling the KV cache via set_kv_prefix().
        """
        token = first_token
        generated = 0
        while token != eos_token_id and generated < max_new_tokens:
            yield token
            generated += 1
            await asyncio.sleep(0)  # cooperate with the event loop (codec/pipecat dispatch)
            token = self.step(token)

    @property
    def position(self) -> int:
        return self._position
