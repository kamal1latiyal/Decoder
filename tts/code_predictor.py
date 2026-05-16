"""
CodePredictor — thin wrapper around the official Qwen3-TTS code predictor
(the "subtalker") sub-model.

Per Qwen/Qwen3-TTS-12Hz-0.6B-Base config.json the code predictor is a 5-layer
qwen3_tts_talker_code_predictor with num_code_groups=16. For each group-0
codec token produced by the talker, it autoregressively predicts groups 1..15
to form a complete 16-codebook RVQ frame.

We do NOT reimplement this — at 5 layers it runs in ~2ms in HF, and
reproducing the exact training-time conditioning would be error-prone.
We just call into the loaded model's `subtalker.generate_one_frame` (the
name used by qwen-tts's modeling code; we feature-detect alternatives).
"""

import torch
from typing import Callable


NUM_CODE_GROUPS = 16


class CodePredictor:
    def __init__(self, qwen3_tts_model: torch.nn.Module):
        """
        Args:
            qwen3_tts_model: a loaded Qwen3TTSForConditionalGeneration instance
                (the full model — we just borrow its subtalker).
        """
        self.device = next(qwen3_tts_model.parameters()).device
        self._fn = self._find_predict_fn(qwen3_tts_model)

    def _find_predict_fn(self, model: torch.nn.Module) -> Callable:
        """Locate the per-frame predictor function on the loaded model.

        We probe a few attribute names because qwen-tts has renamed the
        sub-module across releases (`subtalker`, `code_predictor`,
        `talker_code_predictor`).  All variants share an identical 5-layer
        backbone and a method that takes a group-0 token + the talker's last
        hidden state and returns the 16 RVQ tokens for the current frame.
        """
        for attr in ("subtalker", "code_predictor", "talker_code_predictor"):
            sub = getattr(model, attr, None)
            if sub is None:
                continue
            for method in ("generate_one_frame", "predict_frame", "forward_one_step"):
                fn = getattr(sub, method, None)
                if callable(fn):
                    return fn
        raise AttributeError(
            "Could not locate the code predictor on the loaded Qwen3-TTS model. "
            "Tried submodules: subtalker / code_predictor / talker_code_predictor."
        )

    @torch.inference_mode()
    def predict(self, group0_token: int, talker_hidden: torch.Tensor) -> list[int]:
        """
        Returns one 16-element list [group0, group1, ..., group15] of codec
        token ids.

        Args:
            group0_token: the codec token id emitted by the talker for this frame.
            talker_hidden: [1, hidden_size] — the talker's last-step hidden state,
                conditioning for the subtalker.  Provided by the pipeline.
        """
        out = self._fn(
            group0_token=int(group0_token),
            hidden=talker_hidden.to(self.device),
        )
        # Different versions return either a list, a 1-D tensor, or a tuple.
        if torch.is_tensor(out):
            ids = out.flatten().tolist()
        elif isinstance(out, (list, tuple)):
            ids = list(out)
        else:
            raise TypeError(f"Unexpected predictor output type: {type(out)}")
        if len(ids) != NUM_CODE_GROUPS:
            raise ValueError(f"Expected {NUM_CODE_GROUPS} groups, got {len(ids)}: {ids[:4]}...")
        return [int(x) for x in ids]
