"""
CodePredictor — wraps the Qwen3-TTS talker's code predictor ("subtalker") for
per-frame autoregressive expansion of group-0 → groups 1..15.

Two implementations live here:

  1. `CodePredictor` — HF-driven reference. Calls `code_predictor.generate(...)`
     once per frame. Simple, correct, slow.

  2. `CUDAGraphedCodePredictor` — custom subtalker forward + CUDA graph
     capture. Bypasses HF's Cache/mask/position-id infrastructure entirely
     and only uses the loaded weights as plain tensor projections. The
     captured graph is short (5 layers × 16 positions) and has zero
     CPU-tensor allocations, so torch.cuda.graph() captures it cleanly.

The CUDA-graphed version is what closes the perf gap the rejection feedback
identified: ~40 ms HF subtalker per frame → ~5-10 ms graph replay.

Both classes expose the same predict(group0_token, past_hidden) → (frame_list, codec_hidden_sum).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


NUM_CODE_GROUPS = 16   # 1 talker group-0 + 15 subtalker groups (RVQ)


# ────────────────────────────────────────────────────────────────────────
# Reference implementation (HF-driven, slow but correct)
# ────────────────────────────────────────────────────────────────────────


class CodePredictor:
    """Reference subtalker wrapper using HF `code_predictor.generate(...)`."""

    def __init__(self, qwen3_tts_model: torch.nn.Module):
        talker = qwen3_tts_model.talker
        self._subtalker = talker.code_predictor
        self._group0_embed: torch.nn.Embedding = talker.model.codec_embedding
        self._group_embeds: torch.nn.ModuleList = (
            talker.code_predictor.model.codec_embedding
        )
        if len(self._group_embeds) != NUM_CODE_GROUPS - 1:
            raise ValueError(
                f"Expected {NUM_CODE_GROUPS - 1} sub-group embeddings, got "
                f"{len(self._group_embeds)}"
            )
        self.device = next(talker.parameters()).device
        self.dtype = next(talker.parameters()).dtype

    @torch.inference_mode()
    def predict(
        self,
        group0_token: int,
        past_hidden: torch.Tensor,
        *,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ) -> Tuple[list, torch.Tensor]:
        past_hidden = past_hidden.to(self.device, self.dtype)
        if past_hidden.dim() == 1:
            past_hidden = past_hidden.view(1, 1, -1)
        elif past_hidden.dim() == 2:
            past_hidden = past_hidden.unsqueeze(1) if past_hidden.shape[0] == 1 else past_hidden.unsqueeze(0)

        g0_id = torch.tensor([[group0_token]], dtype=torch.long, device=self.device)
        last_id_hidden = self._group0_embed(g0_id)

        result = self._subtalker.generate(
            inputs_embeds=torch.cat([past_hidden, last_id_hidden], dim=1),
            max_new_tokens=NUM_CODE_GROUPS - 1,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            return_dict_in_generate=False,
            use_cache=True,
        )
        seq = result[:, -((NUM_CODE_GROUPS - 1)):]
        if seq.shape[-1] != NUM_CODE_GROUPS - 1:
            raise RuntimeError(
                f"Subtalker emitted {seq.shape[-1]} tokens, expected {NUM_CODE_GROUPS - 1}"
            )

        parts = [last_id_hidden]
        for i in range(NUM_CODE_GROUPS - 1):
            parts.append(self._group_embeds[i](seq[..., i : i + 1]))
        codec_hidden_sum = torch.cat(parts, dim=1).sum(dim=1, keepdim=True)

        frame = [int(group0_token)] + [int(x) for x in seq.flatten().tolist()]
        return frame, codec_hidden_sum


# ────────────────────────────────────────────────────────────────────────
# Custom subtalker forward + CUDA-graphed predictor
# ────────────────────────────────────────────────────────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half, swap with sign flip on the first half.
    Standard Qwen/Llama-style RoPE convention."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class CUDAGraphedCodePredictor:
    """Subtalker driven by a custom 5-layer transformer forward, wrapped in
    `torch.cuda.CUDAGraph` capture.

    Why custom forward (not HF's `subtalker.model.forward`)?
      HF's forward calls `create_causal_mask`, `get_seq_length`, and a host
      of Cache helpers that internally allocate unpinned CPU tensors. CUDA
      graph capture rejects those. And running 15 forwards/frame through
      that path is ~700× slower than expected (measured on RTX 5090, May 2026)
      because HF's per-call Python prep dominates wall time.

      The custom forward uses the same nn.Linear / RMSNorm modules from the
      loaded subtalker (so the math is identical), but skips ALL of HF's
      orchestration. Per-step ops are: q/k/v projections, q/k_norm, RoPE,
      KV-cache narrow().copy_(), attention (manual matmul + softmax), o_proj,
      MLP. Pure tensor ops. Captures cleanly into a CUDA graph.

    Constraints:
      - Greedy decoding (argmax). Sampling would need an RNG-friendly path.
      - Static shapes throughout. KV cache pre-allocated for the worst-case
        16 positions (1 prefill of 2 + 14 generation steps).
      - First construction does ~3 warmup runs + graph capture (a few hundred
        ms one-time cost). After that, predict() is graph.replay().

    API: same as CodePredictor — predict(group0_token, past_hidden)
         returns (frame_of_16_ints, codec_hidden_sum_bf16[1,1,1024]).
    """

    NUM_SUB_STEPS = NUM_CODE_GROUPS - 1     # 15 generation steps
    PREFILL_LEN = 2                          # past_hidden + g0_embed
    MAX_SEQ_LEN = PREFILL_LEN + NUM_SUB_STEPS - 1   # 16

    def __init__(self, qwen3_tts_model: torch.nn.Module, enable_compile: bool = False):
        """
        Args:
            enable_compile: ignored (kept for backwards-compat with old call
                sites). The CUDA-graph capture path is now the default and
                doesn't go through torch.compile.
        """
        talker = qwen3_tts_model.talker
        sub = talker.code_predictor
        sub_model = sub.model

        # ── Module references (direct projections / norms — no HF wrapping) ──
        self._subtalker = sub
        self._subtalker_model = sub_model
        self._small_to_mtp = sub.small_to_mtp_projection
        self._final_norm = sub_model.norm
        self._lm_heads = sub.lm_head                              # ModuleList of 15
        self._group0_embed: torch.nn.Embedding = talker.model.codec_embedding
        self._group_embeds: torch.nn.ModuleList = sub_model.codec_embedding   # 15
        self._layers = sub_model.layers                            # ModuleList of 5

        if len(self._group_embeds) != self.NUM_SUB_STEPS:
            raise ValueError(
                f"Expected {self.NUM_SUB_STEPS} sub-group embeddings, got {len(self._group_embeds)}"
            )
        if len(self._lm_heads) != self.NUM_SUB_STEPS:
            raise ValueError(f"Expected {self.NUM_SUB_STEPS} lm_heads, got {len(self._lm_heads)}")

        cfg = sub_model.config
        self._num_layers = cfg.num_hidden_layers
        self._num_q_heads = cfg.num_attention_heads
        self._num_kv_heads = cfg.num_key_value_heads
        self._head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self._hidden_size = cfg.hidden_size
        self._kv_repeat = self._num_q_heads // self._num_kv_heads
        self._attn_scale = self._head_dim ** -0.5
        # Subtalker rope_theta — pulled from rotary embedding's config
        self._rope_theta = getattr(cfg, "rope_theta", 1_000_000.0)

        self.device = next(talker.parameters()).device
        self.dtype = next(talker.parameters()).dtype
        self._cpu_test_mode = self.device.type != "cuda"

        H = self._hidden_size
        D = self._head_dim

        # ── Static input buffers (predict() writes these) ──
        self._static_past_hidden = torch.zeros(1, 1, H, device=self.device, dtype=self.dtype)
        self._static_g0_token = torch.zeros(1, 1, device=self.device, dtype=torch.long)

        # ── Static output buffers (predict() reads these) ──
        self._static_out_tokens = torch.zeros(self.NUM_SUB_STEPS, device=self.device, dtype=torch.long)
        self._static_codec_hidden_sum = torch.zeros(1, 1, H, device=self.device, dtype=self.dtype)

        # ── KV cache: [num_layers, 1, num_kv_heads, MAX_SEQ_LEN, head_dim] ──
        # bf16 like HF's default. Slots get overwritten in position order each
        # replay; causal mask ensures we never read stale future slots.
        self._kv_k = torch.zeros(
            self._num_layers, 1, self._num_kv_heads, self.MAX_SEQ_LEN, D,
            device=self.device, dtype=self.dtype,
        )
        self._kv_v = torch.zeros_like(self._kv_k)

        # ── Precomputed RoPE tables for all 16 positions ──
        # Use the subtalker's OWN rotary_emb module — captures any
        # rope_scaling / attention_factor quirks the config carries, instead
        # of re-deriving inv_freq from rope_theta alone.
        rotary = sub_model.rotary_emb
        positions = torch.arange(self.MAX_SEQ_LEN, device=self.device, dtype=torch.long).unsqueeze(0)
        dummy_x = torch.zeros(1, 1, H, device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            cos, sin = rotary(dummy_x, positions)                  # cos/sin: [1, MAX, D]
        self._cos_table = cos.squeeze(0).to(self.dtype)            # [MAX, D]
        self._sin_table = sin.squeeze(0).to(self.dtype)

        # ── Causal mask: for position p, indices > p are -inf, else 0 ──
        # Pre-computed once, indexed per position.
        # Shape: [MAX_SEQ_LEN, MAX_SEQ_LEN] additive mask (broadcast over heads/batch).
        mask = torch.full(
            (self.MAX_SEQ_LEN, self.MAX_SEQ_LEN), float("-inf"),
            device=self.device, dtype=self.dtype,
        )
        # Lower-triangular zero (causal): row p allows attention to columns 0..p.
        mask = torch.triu(mask, diagonal=1)
        self._causal_mask = mask                                  # [MAX, MAX]

        self._rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))

        # ── CUDA Graph state ──
        # TTS_SKIP_GRAPH_CAPTURE=1 → use the custom forward but skip the
        # graph capture step. Useful as a debug ladder: if graph-replay
        # is fast we know the math+capture work; if only the eager custom
        # forward is fast we know the capture path is the issue.
        import os
        skip_capture = os.environ.get("TTS_SKIP_GRAPH_CAPTURE", "0") not in ("0", "false", "False")
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._using_compile = False        # legacy field, false for graph path
        if not self._cpu_test_mode and not skip_capture:
            self._warm_and_capture()
        elif skip_capture and not self._cpu_test_mode:
            print("[CUDAGraph] TTS_SKIP_GRAPH_CAPTURE=1 — graph capture skipped, "
                  "running custom forward eagerly each predict() call.", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Custom per-layer forward — only nn.Linear / RMSNorm calls + tensor ops
    # ──────────────────────────────────────────────────────────────────
    def _layer_forward(self, x: torch.Tensor, layer_idx: int, position: int) -> torch.Tensor:
        """One transformer-layer forward step at a single position.

        Args:
            x:          [1, 1, H]  — input hidden state for this position
            layer_idx:  Python int — which of the 5 layers
            position:   Python int — position in the KV cache (0..MAX_SEQ_LEN-1)

        Writes K/V at slot `position` of this layer's KV cache. Reads K/V
        from slots 0..position-1 (plus the just-written slot) for attention.

        Returns:
            output hidden state, [1, 1, H], with residuals applied.
        """
        layer = self._layers[layer_idx]
        attn = layer.self_attn
        mlp = layer.mlp

        # ── Pre-attention norm ──
        residual = x
        x_ln = layer.input_layernorm(x)                            # [1, 1, H]

        # ── Q/K/V projections ──
        q = attn.q_proj(x_ln)                                      # [1, 1, num_q*D]
        k = attn.k_proj(x_ln)                                      # [1, 1, num_kv*D]
        v = attn.v_proj(x_ln)

        # Reshape to [1, num_heads, 1, head_dim]
        q = q.view(1, 1, self._num_q_heads, self._head_dim).transpose(1, 2)
        k = k.view(1, 1, self._num_kv_heads, self._head_dim).transpose(1, 2)
        v = v.view(1, 1, self._num_kv_heads, self._head_dim).transpose(1, 2)

        # ── Q/K norm (head-dim RMSNorm, only on head_dim) ──
        q = attn.q_norm(q)
        k = attn.k_norm(k)

        # ── RoPE at this position ──
        cos = self._cos_table[position].view(1, 1, 1, self._head_dim)    # [1,1,1,D]
        sin = self._sin_table[position].view(1, 1, 1, self._head_dim)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # ── Write K/V to cache at slot `position` ──
        # narrow() with Python int — no allocation, just stride math.
        self._kv_k[layer_idx, :, :, :, :].narrow(2, position, 1).copy_(k)
        self._kv_v[layer_idx, :, :, :, :].narrow(2, position, 1).copy_(v)

        # ── Attention over the full cache (causal mask handles past-only) ──
        k_full = self._kv_k[layer_idx]                             # [1, num_kv, MAX, D]
        v_full = self._kv_v[layer_idx]
        # GQA: repeat KV heads to match Q heads.
        if self._kv_repeat > 1:
            k_full = k_full.repeat_interleave(self._kv_repeat, dim=1)  # [1, num_q, MAX, D]
            v_full = v_full.repeat_interleave(self._kv_repeat, dim=1)

        # Scores: [1, num_q, 1, D] @ [1, num_q, D, MAX] → [1, num_q, 1, MAX]
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self._attn_scale

        # Apply causal mask for this position (row `position` of the mask).
        # mask[position] is [MAX_SEQ_LEN] with 0 at indices ≤ position, -inf else.
        scores = scores + self._causal_mask[position].view(1, 1, 1, self.MAX_SEQ_LEN)

        attn_probs = scores.softmax(dim=-1)
        # [1, num_q, 1, MAX] @ [1, num_q, MAX, D] → [1, num_q, 1, D]
        attn_out = torch.matmul(attn_probs, v_full)

        # ── Reshape + o_proj ──
        # [1, num_q, 1, D] → [1, 1, num_q*D]
        attn_out = attn_out.transpose(1, 2).reshape(1, 1, self._num_q_heads * self._head_dim)
        attn_out = attn.o_proj(attn_out)

        x = residual + attn_out

        # ── Pre-MLP norm + SwiGLU MLP ──
        residual = x
        x_ln = layer.post_attention_layernorm(x)
        gate = mlp.gate_proj(x_ln)
        up = mlp.up_proj(x_ln)
        mlp_out = mlp.down_proj(F.silu(gate) * up)

        x = residual + mlp_out
        return x

    def _subtalker_step(self, input_hidden: torch.Tensor, position: int) -> torch.Tensor:
        """Run the input through all 5 layers + final norm at `position`.
        Returns the post-norm hidden [1, 1, H]."""
        x = input_hidden
        for layer_idx in range(self._num_layers):
            x = self._layer_forward(x, layer_idx, position)
        x = self._final_norm(x)
        return x

    # ──────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _run_decode_loop(self) -> None:
        """Full 15-step decode using only the static buffers.

        Sequence (matches modeling_qwen3_tts.py talker forward 1665-1692):
          - position 0: feed past_hidden_proj. Discard hidden output (just
            fills KV cache).
          - position 1: feed embed(group0)_proj. Hidden output → lm_head[0]
            → argmax → group-1 token.
          - positions 2..15: feed embed(prev_group)_proj. Hidden output →
            lm_head[step] → argmax → group-(step+1) token.
        """
        # ── Prefill position 0: past_hidden ──
        x0 = self._small_to_mtp(self._static_past_hidden)         # [1, 1, H]
        _ = self._subtalker_step(x0, position=0)                  # discard, KV cache filled

        # ── Prefill position 1: embed(group0) ──
        g0_embed = self._group0_embed(self._static_g0_token)      # [1, 1, H]
        x1 = self._small_to_mtp(g0_embed)
        h1 = self._subtalker_step(x1, position=1)                 # [1, 1, H]
        token_0 = self._lm_heads[0](h1).argmax(dim=-1).squeeze()  # group-1 token
        self._static_out_tokens.narrow(0, 0, 1).copy_(token_0.unsqueeze(0))

        # ── codec_hidden_sum starts with embed(group0) ──
        codec_hidden_sum = g0_embed.clone()

        # ── Generation steps 1..14: one token per step ──
        for step in range(1, self.NUM_SUB_STEPS):
            prev_token = self._static_out_tokens[step - 1].view(1, 1)
            # group_embeds[i] is the embedding for group (i+1); to embed the
            # previously-sampled group-`step` token use group_embeds[step-1].
            step_embed = self._group_embeds[step - 1](prev_token)
            step_proj = self._small_to_mtp(step_embed)
            h = self._subtalker_step(step_proj, position=self.PREFILL_LEN + step - 1)
            token_i = self._lm_heads[step](h).argmax(dim=-1).squeeze()
            self._static_out_tokens.narrow(0, step, 1).copy_(token_i.unsqueeze(0))
            codec_hidden_sum = codec_hidden_sum + step_embed

        # Last group's embedding gets added too (sum spans all 16 groups).
        prev_token = self._static_out_tokens[self.NUM_SUB_STEPS - 1].view(1, 1)
        last_embed = self._group_embeds[self.NUM_SUB_STEPS - 1](prev_token)
        codec_hidden_sum = codec_hidden_sum + last_embed

        self._static_codec_hidden_sum.copy_(codec_hidden_sum)

    # ──────────────────────────────────────────────────────────────────
    def _warm_and_capture(self) -> None:
        """Warm + capture the decode loop into a CUDA graph.

        Why the capture works now (vs the old failed attempt):
          - No HF mask creation (we use pre-computed self._causal_mask).
          - No HF Cache class (we use plain self._kv_k / self._kv_v tensors).
          - No HF position_ids manipulation (we use Python int `position`).
          - All index_copy_/narrow ops use Python ints → stride-only, no
            new tensor allocations during capture.
          - Pre-allocated cos/sin tables, KV cache, input/output buffers.

        The captured graph contains: ~5 layers × ~16 positions × ~6 GPU ops
        per layer = ~480 kernel launches, all bundled into one replay.

        Each phase is logged + timed so that if anything hangs on GPU we can
        tell from the server log exactly which phase wedged (instead of
        guessing from `nvidia-smi 100% util` with no other signal).
        """
        import sys, time

        def _log(msg: str) -> None:
            print(f"[CUDAGraph] {msg}", flush=True)
            sys.stdout.flush()

        # ── Phase 1: bare-eager smoke test (no streams, no capture) ──
        # If this hangs, the custom forward itself has a bug and graph capture
        # was never going to help.
        _log("phase 1/4: eager smoke decode (single _run_decode_loop)...")
        t0 = time.perf_counter()
        self._run_decode_loop()
        torch.cuda.synchronize()
        _log(f"  ✓ eager decode OK ({(time.perf_counter()-t0)*1000:.0f} ms)")

        # ── Phase 2: warmup on side stream — required before cuda graph capture ──
        _log("phase 2/4: warmup on side stream (3 iters)...")
        t0 = time.perf_counter()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                self._run_decode_loop()
                _log(f"  warmup iter {i+1}/3 done")
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        _log(f"  ✓ warmup OK ({(time.perf_counter()-t0)*1000:.0f} ms)")

        # ── Phase 3: CUDA graph capture ──
        _log("phase 3/4: capturing CUDA graph...")
        t0 = time.perf_counter()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._run_decode_loop()
        torch.cuda.synchronize()
        _log(f"  ✓ graph captured ({(time.perf_counter()-t0)*1000:.0f} ms)")

        # ── Phase 4: replay smoke test — confirm the captured graph runs ──
        _log("phase 4/4: replay smoke test...")
        t0 = time.perf_counter()
        self._graph.replay()
        torch.cuda.synchronize()
        _log(f"  ✓ replay OK ({(time.perf_counter()-t0)*1000:.0f} ms)")
        _log("ready — per-frame decode will use captured graph.")

    # ──────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict(
        self,
        group0_token: int,
        past_hidden: torch.Tensor,
        **kwargs,
    ) -> Tuple[list, torch.Tensor]:
        """Drop-in replacement for `CodePredictor.predict()`. **kwargs
        accepted for signature compat (do_sample/top_k/top_p/temperature)
        but ignored — this implementation is greedy."""
        # Stage inputs into static buffers.
        if past_hidden.dim() == 1:
            past_hidden = past_hidden.view(1, 1, -1)
        elif past_hidden.dim() == 2:
            past_hidden = past_hidden.unsqueeze(1) if past_hidden.shape[0] == 1 else past_hidden.unsqueeze(0)
        self._static_past_hidden.copy_(past_hidden.to(self.device, self.dtype))
        self._static_g0_token.fill_(int(group0_token))

        if self._graph is not None:
            # Fast path: replay the captured CUDA graph.
            self._graph.replay()
            torch.cuda.synchronize()
        else:
            # CPU test mode: run the decode loop directly. Same numerical path.
            self._run_decode_loop()

        # Read outputs.
        frame_tail = self._static_out_tokens.tolist()
        frame = [int(group0_token)] + [int(x) for x in frame_tail]
        # Clone codec_hidden_sum — caller needs a stable copy (the static
        # buffer gets overwritten on the next predict() call).
        codec_hidden_sum = self._static_codec_hidden_sum.clone()

        return frame, codec_hidden_sum
