"""
CodePredictor — wraps the Qwen3-TTS talker's code predictor ("subtalker") for
per-frame autoregressive expansion of group-0 → groups 1..15.

From qwen_tts/core/models/modeling_qwen3_tts.py (verified against package
version 0.1.x), the subtalker lives at:
    model.talker.code_predictor   →  Qwen3TTSTalkerCodePredictorModelForConditionalGeneration

Its per-step generate signature (matching how the talker itself drives it
inside Qwen3TTSTalkerForConditionalGeneration.forward, lines 1671–1680):

    predictor_result = code_predictor.generate(
        inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),   # [B, 2, 1024]
        max_new_tokens=15,                  # groups 1..15 (group 0 = talker output)
        do_sample=subtalker_dosample,
        top_p=subtalker_top_p,
        top_k=subtalker_top_k,
        temperature=subtalker_temperature,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    predictor_result.sequences           # [B, 15] codec ids for groups 1..15

Where:
  past_hidden     : talker's last-step POST-norm hidden state (shape [B, 1, 1024])
                    — for us, comes from MegakernelDecoder.step_from_hidden's
                    second return value, broadcast to [1, 1, 1024].
  last_id_hidden  : talker's codec_embedding(group0_token).unsqueeze(0)
                    shape [1, 1, 1024]

CodePredictor.predict() also returns the **summed codec embedding** so the
pipeline can feed it (plus trailing_text_hidden) back into the megakernel
as next-step input — replicating exactly what the talker does internally
at lines 1683–1687 of the modeling file.
"""

from typing import Tuple

import torch


NUM_CODE_GROUPS = 16   # 1 talker group-0 + 15 subtalker groups (RVQ)


class CodePredictor:
    """Wraps the subtalker for one-step-at-a-time generation."""

    def __init__(self, qwen3_tts_model: torch.nn.Module):
        """
        Args:
            qwen3_tts_model: a loaded Qwen3TTSForConditionalGeneration instance.
                We borrow:
                  - .talker.code_predictor          (the subtalker module)
                  - .talker.model.codec_embedding   (group-0 codec embed)
                  - .talker.code_predictor.model.codec_embedding
                                                    (group 1..15 codec embeds,
                                                     as a nn.ModuleList of 15)
        """
        talker = qwen3_tts_model.talker
        self._subtalker = talker.code_predictor
        # Group-0 embedding (talker side, shared codec vocab table).
        self._group0_embed: torch.nn.Embedding = talker.model.codec_embedding
        # Groups 1..15 embeddings (subtalker side, one nn.Embedding per group).
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
    ) -> Tuple[list[int], torch.Tensor]:
        """
        For one frame, expand the talker's group-0 token into a full 16-group
        codec frame and return the embedding-sum the talker uses as its
        next-step input.

        Args:
            group0_token: codec id from the talker (group 0).
            past_hidden:  the talker's post-norm hidden state from the same
                          step that produced `group0_token`. Shape [1, 1, 1024]
                          or [1024] (we reshape).
            do_sample/top_k/top_p/temperature:
                          subtalker sampling controls; defaults match qwen_tts's
                          generate_voice_clone defaults.

        Returns:
            (frame, codec_hidden_sum)
              frame:            list[int] of length 16 = [g0, g1, ..., g15]
              codec_hidden_sum: bf16 tensor [1, 1, 1024]
                                = embed(g0) + Σᵢ₌₁..₁₅ embedᵢ(gᵢ)
                                  (the "codec_hiddens.sum(1)" the talker uses
                                   to construct its NEXT-step input embedding)
        """
        past_hidden = past_hidden.to(self.device, self.dtype)
        if past_hidden.dim() == 1:
            past_hidden = past_hidden.view(1, 1, -1)
        elif past_hidden.dim() == 2:
            past_hidden = past_hidden.unsqueeze(1) if past_hidden.shape[0] == 1 else past_hidden.unsqueeze(0)

        # group-0 embedding (from the talker's codec embed table).
        g0_id = torch.tensor([[group0_token]], dtype=torch.long, device=self.device)
        last_id_hidden = self._group0_embed(g0_id)                  # [1, 1, 1024]

        # Run the subtalker for 15 steps to emit groups 1..15.
        # NB: output_hidden_states was True in an earlier version — that forces
        # HF's GenerationMixin to collect 5 layers × 15 steps = 75 hidden-state
        # tensors per frame which we never use (the kernel exposes its own
        # post-norm hidden via MegakernelDecoder.norm_out). At 12.5 frames/sec
        # that was ~940 unnecessary allocations/sec and dominated wall time.
        # We only need `.sequences`, so skip the dict and the hidden-state plumbing.
        result = self._subtalker.generate(
            inputs_embeds=torch.cat([past_hidden, last_id_hidden], dim=1),  # [1, 2, 1024]
            max_new_tokens=NUM_CODE_GROUPS - 1,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            return_dict_in_generate=False,   # plain Tensor return, no dict
            use_cache=True,                  # explicit (default-ish, document intent)
        )
        # When return_dict_in_generate=False, generate() returns the full id tensor
        # including the prompt prefix. We sliced it to keep only the new tokens.
        # inputs_embeds prefix length = 2 (past_hidden + last_id_hidden); the
        # subtalker emits 15 new ids after that.
        seq = result[:, -((NUM_CODE_GROUPS - 1)):]                   # [1, 15]
        if seq.shape[-1] != NUM_CODE_GROUPS - 1:
            raise RuntimeError(
                f"Subtalker emitted {seq.shape[-1]} tokens, expected "
                f"{NUM_CODE_GROUPS - 1}"
            )

        # Build the codec_hidden sum the talker uses for the NEXT step's input
        # (modeling_qwen3_tts.py lines 1683–1687 — verbatim layout):
        #   codec_hiddens = [embed(g0), sub_embed[0](g1), ..., sub_embed[14](g15)]
        #   inputs_embeds = codec_hiddens.sum(1, keepdim=True)
        parts = [last_id_hidden]
        for i in range(NUM_CODE_GROUPS - 1):
            parts.append(self._group_embeds[i](seq[..., i : i + 1]))   # [1, 1, 1024]
        codec_hidden_sum = torch.cat(parts, dim=1).sum(dim=1, keepdim=True)  # [1, 1, 1024]

        frame = [int(group0_token)] + [int(x) for x in seq.flatten().tolist()]
        return frame, codec_hidden_sum
