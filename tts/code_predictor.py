"""
CodePredictor — wraps the Qwen3-TTS talker's code predictor ("subtalker") for
per-frame autoregressive expansion of group-0 → groups 1..15.

Two implementations live in this file:

  1. `CodePredictor`  — the original HF-driven version. Calls
     `code_predictor.generate(...)` once per frame. Simple, correct, but
     each call pays ~30-50 ms of Python overhead in HF's `GenerationMixin`
     (cache init, argument parsing, hook dispatch, position-id maths,
     per-step preparation). At 12.5 frames/sec this dominates wall time.

  2. `CUDAGraphedCodePredictor` — the optimised version. Replaces
     `code_predictor.generate` with a manual 15-step decode loop using
     transformers' `StaticCache` (CUDA-graph-compatible) and wraps the
     entire loop in a `torch.cuda.CUDAGraph` capture/replay. Each frame
     becomes a single ~10 µs `graph.replay()` call instead of 15 HF
     forward calls + their Python boilerplate.

Pick which one via the `use_cuda_graph` flag on TTSPipeline. Default is
graphed; setting it to False falls back to the HF version for A/B.

Both classes have the same `.predict(group0_token, past_hidden) -> (frame_list, codec_hidden_sum)`
signature so the pipeline orchestrator doesn't care which is loaded.

Background on the per-step contract (verified against modeling_qwen3_tts.py
lines 1671–1692 in qwen_tts 0.1.x):
  past_hidden     : talker's POST-norm hidden state, [1, 1, 1024]
                    (from MegakernelDecoder.step_from_hidden's 2nd return)
  group0_token    : the talker's next codec emission (an int)
  ↓
  subtalker prefill on [past_hidden, embed(group0_token)] → 14 more
  generation steps using the subtalker's per-group sub_embeds + lm_heads
  ↓
  returns 16-element frame (group 0 + groups 1..15) plus the SUMMED
  codec embedding that the talker uses as its NEXT-step input.
"""

from typing import Optional, Tuple

import torch


NUM_CODE_GROUPS = 16   # 1 talker group-0 + 15 subtalker groups (RVQ)


# ────────────────────────────────────────────────────────────────────────
# Reference implementation — uses HF GenerationMixin (slow, no graph)
# ────────────────────────────────────────────────────────────────────────


class CodePredictor:
    """Reference subtalker wrapper using HF `code_predictor.generate(...)`.

    Slower path; kept as the correctness baseline and as a fallback when
    CUDA graphs aren't available (e.g. CPU sanity tests).
    """

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
        last_id_hidden = self._group0_embed(g0_id)                       # [1, 1, 1024]

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
        seq = result[:, -((NUM_CODE_GROUPS - 1)):]                       # [1, 15]
        if seq.shape[-1] != NUM_CODE_GROUPS - 1:
            raise RuntimeError(
                f"Subtalker emitted {seq.shape[-1]} tokens, expected {NUM_CODE_GROUPS - 1}"
            )

        parts = [last_id_hidden]
        for i in range(NUM_CODE_GROUPS - 1):
            parts.append(self._group_embeds[i](seq[..., i : i + 1]))      # [1, 1, 1024]
        codec_hidden_sum = torch.cat(parts, dim=1).sum(dim=1, keepdim=True)

        frame = [int(group0_token)] + [int(x) for x in seq.flatten().tolist()]
        return frame, codec_hidden_sum


# ────────────────────────────────────────────────────────────────────────
# Fast path — manual loop + CUDA graph capture/replay
# ────────────────────────────────────────────────────────────────────────


class CUDAGraphedCodePredictor:
    """Subtalker driven by a manual 15-step decode loop captured into a
    `torch.cuda.CUDAGraph`. Per-frame cost drops from ~40 ms (HF generate
    overhead × 15 sub-steps) to ~3-10 ms (one graph replay).

    Constraints:
      - Greedy (argmax) only. CUDA graphs need deterministic control flow,
        and HF's sampling pulls random numbers each call which is awkward
        to capture. Greedy also matches the megakernel's argmax, which is
        the spirit of the kernel anyway.
      - Static shapes throughout. Pre-allocates input/output buffers + a
        `StaticCache` sized for the worst case (16 positions).
      - First call after construction does the warmup + capture. Subsequent
        calls are pure graph.replay() + tiny input/output copies.

    API: same as CodePredictor — `.predict(group0_token, past_hidden)`
    returns `(frame_list_of_16_ints, codec_hidden_sum [1,1,1024] bf16)`.
    """

    NUM_GROUPS = NUM_CODE_GROUPS                # 16
    NUM_SUB_STEPS = NUM_CODE_GROUPS - 1         # 15 generation steps
    PREFILL_LEN = 2                             # past_hidden + g0_embed
    MAX_CACHE_LEN = PREFILL_LEN + NUM_SUB_STEPS - 1   # 16 — worst-case KV positions

    def __init__(self, qwen3_tts_model: torch.nn.Module):
        from transformers.cache_utils import StaticCache

        talker = qwen3_tts_model.talker
        self._subtalker = talker.code_predictor
        self._subtalker_model = talker.code_predictor.model
        self._lm_heads = talker.code_predictor.lm_head            # ModuleList of 15
        self._small_to_mtp = talker.code_predictor.small_to_mtp_projection
        self._group0_embed: torch.nn.Embedding = talker.model.codec_embedding
        self._group_embeds: torch.nn.ModuleList = (                # 15 nn.Embeddings
            talker.code_predictor.model.codec_embedding
        )
        if len(self._group_embeds) != self.NUM_SUB_STEPS:
            raise ValueError(
                f"Expected {self.NUM_SUB_STEPS} sub-group embeddings, got "
                f"{len(self._group_embeds)}"
            )
        if len(self._lm_heads) != self.NUM_SUB_STEPS:
            raise ValueError(
                f"Expected {self.NUM_SUB_STEPS} lm_heads, got {len(self._lm_heads)}"
            )

        self.device = next(talker.parameters()).device
        self.dtype = next(talker.parameters()).dtype
        if self.device.type != "cuda":
            raise RuntimeError(
                "CUDAGraphedCodePredictor requires CUDA. Use CodePredictor on CPU."
            )

        H = self._subtalker.config.hidden_size

        # ── static input buffers (written by predict() each call) ──
        self._static_past_hidden = torch.zeros(1, 1, H, device=self.device, dtype=self.dtype)
        self._static_g0_token = torch.zeros(1, 1, device=self.device, dtype=torch.long)

        # ── static output buffers (read by predict() after replay) ──
        self._static_out_tokens = torch.zeros(self.NUM_SUB_STEPS, device=self.device, dtype=torch.long)
        self._static_codec_hidden_sum = torch.zeros(1, 1, H, device=self.device, dtype=self.dtype)

        # ── static KV cache (worst case 16 positions, all 5 layers) ──
        # StaticCache.__init__ signature changed across transformers versions; try
        # both forms with a small adapter.
        cache_config = self._subtalker_model.config
        try:
            self._cache = StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=self.MAX_CACHE_LEN,
                device=self.device,
                dtype=self.dtype,
            )
        except TypeError:
            # older signature
            self._cache = StaticCache(
                config=cache_config,
                batch_size=1,
                max_cache_len=self.MAX_CACHE_LEN,
                device=self.device,
                dtype=self.dtype,
            )

        # Precompute the per-step `cache_position` tensors — static, so they
        # live on-device and aren't re-allocated each replay.
        self._cache_pos_prefill = torch.arange(
            self.PREFILL_LEN, device=self.device, dtype=torch.long
        )
        self._cache_pos_step = [
            torch.tensor([self.PREFILL_LEN + i - 1], device=self.device, dtype=torch.long)
            for i in range(1, self.NUM_SUB_STEPS)
        ]

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._warm_and_capture()

    # ──────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _run_decode_loop(self) -> None:
        """Manual 15-step decode using only the static buffers.

        Matches the talker's per-step contract in modeling_qwen3_tts.py:
          - Step 0 prefill: 2-token input [past_hidden, embed(g0)] → through
            small_to_mtp_projection → subtalker.model.forward → hidden at
            position 1 → lm_head[0] → group-1 token (greedy argmax).
          - Steps 1..14: 1-token input = embed of prev group-`i` token, via
            sub_embeds[i-1] → small_to_mtp_projection → forward at position
            `i+1` → lm_head[i] → group-(i+1) token.

        We also accumulate `codec_hidden_sum` = embed(g0) + Σᵢ₌₁..₁₅ embedᵢ(gᵢ)
        — the talker's next-step input embedding (sum of all 16 group embeds).
        """
        # Embed g0 (talker side codec embedding — the [3072, 1024] table).
        g0_embed = self._group0_embed(self._static_g0_token)                   # [1, 1, 1024]

        # Two-token prefill input: [past_hidden_proj, g0_embed_proj].
        first_inputs = torch.cat([self._static_past_hidden, g0_embed], dim=1)  # [1, 2, 1024]
        first_inputs_proj = self._small_to_mtp(first_inputs)

        out = self._subtalker_model(
            inputs_embeds=first_inputs_proj,
            past_key_values=self._cache,
            cache_position=self._cache_pos_prefill,
            position_ids=self._cache_pos_prefill.unsqueeze(0),
            use_cache=True,
        )
        # Take the hidden at position 1 (the g0 slot, second of two prefill positions).
        hidden = out.last_hidden_state[:, -1:, :]                              # [1, 1, 1024]
        token_0 = self._lm_heads[0](hidden).argmax(dim=-1).squeeze()           # group-1 token id
        # Use index_copy_ so the write is captured as a static op (not a
        # Python-side assignment which CUDA graphs can't see).
        self._static_out_tokens.index_copy_(
            0, torch.tensor([0], device=self.device, dtype=torch.long), token_0.unsqueeze(0)
        )

        # Initialise the codec_hidden_sum with embed(g0) (talker's group-0 embed).
        codec_hidden_sum = g0_embed.clone()                                    # [1, 1, 1024]

        # Steps 1..14: 14 more generation steps, single token each.
        for step in range(1, self.NUM_SUB_STEPS):
            prev_token = self._static_out_tokens[step - 1].view(1, 1)          # [1, 1]
            # sub_embeds[i] is the embedding for group (i+1) tokens, so to
            # embed the previously-sampled group-`step` token we use
            # sub_embeds[step - 1].
            step_embed = self._group_embeds[step - 1](prev_token)              # [1, 1, 1024]
            step_inputs_proj = self._small_to_mtp(step_embed)

            cache_pos = self._cache_pos_step[step - 1]
            out = self._subtalker_model(
                inputs_embeds=step_inputs_proj,
                past_key_values=self._cache,
                cache_position=cache_pos,
                position_ids=cache_pos.unsqueeze(0),
                use_cache=True,
            )
            hidden = out.last_hidden_state                                     # [1, 1, 1024]
            token_i = self._lm_heads[step](hidden).argmax(dim=-1).squeeze()
            self._static_out_tokens.index_copy_(
                0, torch.tensor([step], device=self.device, dtype=torch.long), token_i.unsqueeze(0)
            )

            # Accumulate group-`step` embedding into the codec_hidden_sum.
            codec_hidden_sum = codec_hidden_sum + step_embed

        # Note we DON'T add the very last token's embedding — that token is
        # group-15 and we just sampled it; the talker's next-step input
        # equation sums embeds of group-0 .. group-15 from the CURRENT frame,
        # so all 16 should be in the sum. Add the final one outside the loop.
        prev_token = self._static_out_tokens[self.NUM_SUB_STEPS - 1].view(1, 1)
        last_embed = self._group_embeds[self.NUM_SUB_STEPS - 1](prev_token)
        codec_hidden_sum = codec_hidden_sum + last_embed

        # Write the final sum into the static output buffer (in-place copy
        # so the address is stable across replays).
        self._static_codec_hidden_sum.copy_(codec_hidden_sum)

    # ──────────────────────────────────────────────────────────────────
    def _warm_and_capture(self) -> None:
        """Warmup (3 eager runs to materialise workspaces) + capture.

        StaticCache slots get overwritten naturally each replay because we
        always write positions 0..N in order. Causal attention only reads
        positions ≤ current step, all of which were just written this
        replay. So no explicit cache reset is needed.
        """
        # Warmup on a side stream — the canonical PyTorch idiom for CUDA-graph
        # capture. Ensures all lazy allocations (workspace buffers in cuBLAS,
        # etc.) happen before capture freezes the memory pool.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._run_decode_loop()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._run_decode_loop()

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

        # Replay the graph — single CUDA submission, no Python in the hot path.
        self._graph.replay()

        # Sync so we can read the int outputs back.
        torch.cuda.synchronize()

        # Build the frame list: [group0_token] + the 15 sampled subtalker tokens.
        frame_tail = self._static_out_tokens.tolist()
        frame = [int(group0_token)] + [int(x) for x in frame_tail]

        # Clone the codec_hidden_sum so the caller gets a stable copy (the
        # static buffer gets overwritten on the next predict() call).
        codec_hidden_sum = self._static_codec_hidden_sum.clone()

        return frame, codec_hidden_sum
