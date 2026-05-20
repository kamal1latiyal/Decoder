"""
MegakernelDecoder — wraps the qwen_megakernel CUDA op for a single codec
autoregressive decode step of the Qwen3-TTS talker.

WHY THIS IS NOT A DROP-IN OF THE UPSTREAM QWEN3-0.6B DECODER
============================================================
Upstream qwen_megakernel drives Qwen3-0.6B: per-step input is a single token
id (which the kernel looks up via `embed_weight + token_id * HIDDEN_SIZE`).

The Qwen3-TTS talker's per-step input is NOT a single token. From
modeling_qwen3_tts.py (Qwen3TTSTalkerForConditionalGeneration.forward, lines
1665–1692), the talker's per-step input embedding is:

    inputs_embeds = Σᵢ embedᵢ(groupᵢ) + trailing_text_hidden[step]

i.e. the SUM of 16 codec-token embeddings (one per RVQ codebook) plus a
projected text-hidden vector for the current step. This is a 1024-d bf16
vector that cannot be represented as embed_weight[token_id] for any token_id.

SOLUTION (no kernel modification needed)
----------------------------------------
The kernel does `embed_row = embed_weight + input_token_id * HIDDEN_SIZE`
(csrc/kernel.cu:1242, 1359). If we pass `embed_weight = <a 1×1024 bf16
tensor whose row 0 is our pre-computed inputs_embeds>` and `token_id = 0`,
the kernel reads our row 0 as the layer-0 input. The CUDA code path is
identical to the normal embed lookup — same memory traffic, same address
calculation, same downstream layers. There is no semantic difference from
"directly feeding a hidden vector"; we are just abusing the embed table as
a 1-entry hidden cache.

POST-NORM HIDDEN STATE EXPOSURE
--------------------------------
The kernel writes the post-norm hidden state into the `g_normalized` buffer
(float32 [HIDDEN_SIZE]) before applying the LM head. We allocate that buffer
as `self.norm_out` and return a bf16 view of it after each step. The talker's
subtalker takes `past_hidden = talker.last_hidden_state[:, -1:]`, which is
exactly this post-norm vector — so we feed it directly to CodePredictor on
the NEXT step. This fixes the "subtalker conditioning is one-step stale"
limitation that the initial DESIGN doc listed.

RoPE / MRoPE
------------
The kernel uses standard 1D RoPE. The talker config specifies MRoPE with
mrope_section=[24,20,20] (separate scales for temporal / height / width
position dims). For single-channel speech (no image/video axis) all three
position dims advance together by 1 each step, so MRoPE degenerates to
ordinary RoPE. We use theta=1_000_000 (talker's `rope_theta`).
"""

import asyncio
import math
from typing import AsyncGenerator, Optional, Tuple

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
SCRATCH_BLOCKS = 4096                # matches upstream Decoder; holds worst-case LM head blocks


class MegakernelDecoder:
    """
    Stateful single-step decoder, calling
    `torch.ops.qwen_megakernel_C.decode(...)` for each codec token.

    Primary API for Qwen3-TTS:
        step_from_hidden(input_hidden_bf16) -> (next_token_id, post_norm_hidden_bf16)

    Secondary (Qwen3-0.6B-style) API, kept for sanity checks / smoke tests:
        step(token_id) -> next_token_id   — kernel does embed lookup itself
    """

    def __init__(self, talker_kernel_weights: TalkerKernelWeights):
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
        self.norm_out = torch.empty(HIDDEN_SIZE, **f32)   # ← kernel writes post-norm here
        self.bmax_vals = torch.empty(SCRATCH_BLOCKS, **f32)
        self.bmax_idxs = torch.empty(SCRATCH_BLOCKS, dtype=torch.int32, device=self.device)
        self.out_token = torch.empty(1, dtype=torch.int32, device=self.device)

        # Single-row embed table used by step_from_hidden().  Row 0 is overwritten
        # each step with the caller's input hidden; kernel is invoked with
        # token_id=0 so it reads row 0 as the layer-0 input.
        self._single_row_embed = torch.zeros(1, HIDDEN_SIZE, **bf16)

    def _build_rope_tables(self) -> None:
        """RoPE cos/sin with theta=1_000_000 (Qwen3-TTS talker)."""
        positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        freqs = torch.outer(positions, inv_freq)            # [MAX, HEAD/2]
        # Upstream uses `.repeat(1, 2)` (concat-like for split-half RoPE,
        # which matches `rotate_half(x) = cat([-x2, x1])`).
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

        Expected shape:
          k_prefix, v_prefix: [NUM_LAYERS, NUM_KV_HEADS, prefix_len, HEAD_DIM]
          dtype: bfloat16

        After this call, the next decode step will continue from
        position = prefix_len.
        """
        if k_prefix.dim() != 4:
            raise ValueError(f"k_prefix must be 4D; got {k_prefix.shape}")
        if k_prefix.shape != v_prefix.shape:
            raise ValueError("k_prefix and v_prefix shapes differ")
        if (
            k_prefix.shape[0] != NUM_LAYERS
            or k_prefix.shape[1] != NUM_KV_HEADS
            or k_prefix.shape[3] != HEAD_DIM
        ):
            raise ValueError(
                f"k_prefix shape {tuple(k_prefix.shape)} must be "
                f"[{NUM_LAYERS}, {NUM_KV_HEADS}, prefix_len, {HEAD_DIM}]"
            )
        prefix_len = k_prefix.shape[2]
        if prefix_len > MAX_SEQ_LEN:
            raise ValueError(f"KV prefix len {prefix_len} > MAX_SEQ_LEN {MAX_SEQ_LEN}")

        self.k_cache[:, :, :prefix_len, :].copy_(k_prefix.to(self.device, torch.bfloat16))
        self.v_cache[:, :, :prefix_len, :].copy_(v_prefix.to(self.device, torch.bfloat16))
        self._position = prefix_len

    # ------------------------------------------------------------------
    # decode primitives
    # ------------------------------------------------------------------

    def step_from_hidden(self, input_hidden: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Single decode step that takes a pre-computed 1024-d input hidden
        vector instead of a token id.  Returns the next codec token id AND
        a bf16 view of the post-norm hidden state (shape [HIDDEN_SIZE]).

        The kernel does an embed lookup `embed_weight + token_id * HIDDEN_SIZE`.
        We pass a 1-row embed table containing our `input_hidden` and
        `token_id=0`, so the lookup lands on our injected vector.

        Args:
            input_hidden: bf16 tensor, shape [HIDDEN_SIZE] or [..., HIDDEN_SIZE]
                          (anything with HIDDEN_SIZE trailing dim is flattened).

        Returns:
            (next_token_id, post_norm_hidden_bf16[HIDDEN_SIZE])
        """
        if self._position >= MAX_SEQ_LEN:
            raise RuntimeError(f"Sequence exceeded MAX_SEQ_LEN={MAX_SEQ_LEN}")
        if input_hidden.numel() != HIDDEN_SIZE:
            raise ValueError(
                f"input_hidden must have {HIDDEN_SIZE} elements; got {input_hidden.numel()}"
            )

        # Stash the input into row 0 of our single-row table.  No allocation,
        # just a copy of 1024 * 2 = 2048 bytes.
        self._single_row_embed[0].copy_(
            input_hidden.detach().to(self.device, torch.bfloat16).reshape(HIDDEN_SIZE)
        )

        self._decode(
            self.out_token,
            0,                                       # token_id = 0 → looks up row 0
            self._single_row_embed,                  # 1×1024 bf16, row 0 = our hidden
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

        # Take the post-norm hidden the kernel just wrote (float32) and
        # cast to bf16 for the subtalker.  Detach from any autograd graph
        # (there shouldn't be one, but be safe under torch.inference_mode).
        post_norm = self.norm_out.to(torch.bfloat16)
        next_token = int(self.out_token.item())  # ~1 µs GPU→CPU sync; required to drive loop
        return next_token, post_norm

    def step(self, token_id: int) -> int:
        """
        Legacy / Qwen3-0.6B-style single-token decode step.
        Kernel does its own embed lookup against the real codec embed table.

        Useful for the smoke-test "cold decode" sanity stage; not used by
        the Qwen3-TTS pipeline (the talker's per-step input is not a single
        token id — see step_from_hidden).
        """
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
        return int(self.out_token.item())

    @property
    def position(self) -> int:
        return self._position
