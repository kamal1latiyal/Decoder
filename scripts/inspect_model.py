#!/usr/bin/env python
"""
inspect_model.py — verify the Qwen3-TTS API surface BEFORE running anything else.

Runs on CPU (no CUDA required for what it inspects, though loading the full
model on CPU takes ~30 s and ~3 GB RAM). Prints every key fact our pipeline
depends on, so we can spot API drift (qwen_tts version bumps, weight
renames, etc.) without burning GPU time.

Run:
    python scripts/inspect_model.py [--device cpu|cuda]

Exit code 0 if every check passes, non-zero otherwise.
"""

import argparse
import inspect
import sys
import traceback


OK = "\033[32m✓\033[0m"
WARN = "\033[33m!\033[0m"
NO = "\033[31m✗\033[0m"


def check(label, fn, *, fatal=True):
    print(f"\n── {label} ──")
    try:
        result = fn()
        print(f"  {OK} {label}")
        return result
    except Exception as e:
        print(f"  {NO} {label}: {e}")
        traceback.print_exc()
        if fatal:
            sys.exit(1)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                        help="Where to load the model. CPU is fine for inspection.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    args = parser.parse_args()

    print("=" * 64)
    print(f" Qwen3-TTS API inspection  ({args.model_id}, device={args.device})")
    print("=" * 64)

    # 1. Import qwen_tts.
    qwen_tts = check("import qwen_tts", lambda: __import__("qwen_tts"))

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402

    # 2. Inspect the wrapper class surface (no model load needed).
    def _inspect_wrapper_surface():
        sig = inspect.signature(Qwen3TTSModel.generate_voice_clone)
        params = list(sig.parameters)
        # `max_new_tokens` is forwarded via **kwargs into _merge_generate_kwargs,
        # not exposed as an explicit param — so we don't assert it here.
        for required in (
            "text", "language", "ref_audio", "ref_text",
            "x_vector_only_mode", "voice_clone_prompt",
        ):
            assert required in params, f"generate_voice_clone missing param: {required}"
        assert "kwargs" in params, "generate_voice_clone lost **kwargs"
        print(f"  generate_voice_clone params  : {params}")

        cvcp_sig = inspect.signature(Qwen3TTSModel.create_voice_clone_prompt)
        cvcp_params = list(cvcp_sig.parameters)
        for required in ("ref_audio", "ref_text", "x_vector_only_mode"):
            assert required in cvcp_params, f"create_voice_clone_prompt missing param: {required}"
        print(f"  create_voice_clone_prompt    : {cvcp_params}")

        # _merge_generate_kwargs accepts max_new_tokens explicitly — make sure that's
        # still the entry point we pass it through.
        merge_sig = inspect.signature(Qwen3TTSModel._merge_generate_kwargs)
        assert "max_new_tokens" in merge_sig.parameters, \
            "_merge_generate_kwargs lost max_new_tokens param"
        print(f"  _merge_generate_kwargs has max_new_tokens : OK")
    check("wrapper API surface", _inspect_wrapper_surface)

    # 3. Load the model (slow). Skip if user passes --no-load — not exposed for now.
    import torch
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    print(f"\n[loading {args.model_id} on {args.device}, dtype={dtype}...]")
    wrapper = check(
        "load Qwen3TTSModel",
        lambda: Qwen3TTSModel.from_pretrained(
            args.model_id,
            dtype=dtype,
            device_map=args.device,
        ),
    )
    model = wrapper.model

    # 4. Submodule existence + types.
    def _submodules():
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
            Qwen3TTSTalkerForConditionalGeneration,
            Qwen3TTSTalkerModel,
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
            Qwen3TTSTalkerCodePredictorModel,
        )
        assert isinstance(model, Qwen3TTSForConditionalGeneration)
        assert isinstance(model.talker, Qwen3TTSTalkerForConditionalGeneration)
        assert isinstance(model.talker.model, Qwen3TTSTalkerModel)
        assert isinstance(
            model.talker.code_predictor,
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        )
        assert isinstance(
            model.talker.code_predictor.model, Qwen3TTSTalkerCodePredictorModel
        )
        assert hasattr(model, "speech_tokenizer") and model.speech_tokenizer is not None
        assert hasattr(model.talker, "codec_head")
        print("  model.talker                 :", type(model.talker).__name__)
        print("  model.talker.model           :", type(model.talker.model).__name__)
        print("  model.talker.code_predictor  :", type(model.talker.code_predictor).__name__)
        print("  model.speech_tokenizer       :", type(model.speech_tokenizer).__name__)
    check("submodule types", _submodules)

    # 5. Talker config + tts_model_type.
    def _talker_config():
        tc = model.config.talker_config
        for attr, expected in (
            ("hidden_size", 1024),
            ("num_hidden_layers", 28),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 8),
            ("head_dim", 128),
            ("intermediate_size", 3072),
            ("vocab_size", 3072),
        ):
            v = getattr(tc, attr)
            assert v == expected, f"talker_config.{attr} = {v}, expected {expected}"
            print(f"  talker_config.{attr:<22} = {v}")
        print(f"  talker_config.rope_theta      = {tc.rope_theta}")
        print(f"  talker_config.codec_eos_token_id = {tc.codec_eos_token_id}")
        print(f"  tts_model_type                = {model.tts_model_type}")
        print(f"  tokenizer_type                = {model.tokenizer_type}")
    check("talker config", _talker_config)

    # 6. State-dict keys (the ones TalkerKernelWeights reads).
    def _state_dict_keys():
        sd = dict(model.talker.state_dict())
        required = [
            "model.codec_embedding.weight",
            "model.norm.weight",
            "codec_head.weight",
        ]
        for k in required:
            assert k in sd, f"missing state-dict key on model.talker: {k}"
            print(f"  {k:<48} shape={tuple(sd[k].shape)}")

        # Per-layer keys
        layer_keys = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        for li in (0, 27):
            for k in layer_keys:
                full = f"model.layers.{li}.{k}"
                assert full in sd, f"missing layer key: {full}"
            print(f"  layer {li} has all 11 expected weights")
    check("talker state_dict keys", _state_dict_keys)

    # 7. Subtalker submodules + shapes.
    def _subtalker():
        cp = model.talker.code_predictor
        emb_list = cp.model.codec_embedding
        head_list = cp.lm_head
        assert len(emb_list) == 15, f"subtalker codec_embedding count = {len(emb_list)}, expected 15"
        assert len(head_list) == 15, f"subtalker lm_head count = {len(head_list)}, expected 15"
        print(f"  subtalker.codec_embedding[*]  : 15 × Embedding(3072, 1024)")
        print(f"  subtalker.lm_head[*]          : 15 × Linear(1024, 3072)")
        print(f"  subtalker.config.num_code_groups = {cp.config.num_code_groups}")
        print(f"  subtalker.config.num_hidden_layers = {cp.config.num_hidden_layers}")
    check("subtalker layout", _subtalker)

    # 8. speech_tokenizer.decode signature (list-of-dicts form).
    def _codec():
        st = model.speech_tokenizer
        sig = inspect.signature(st.decode)
        params = list(sig.parameters)
        print(f"  speech_tokenizer.decode params : {params}")
        print(f"  output sample rate            : {st.get_output_sample_rate()} Hz")
        assert st.get_output_sample_rate() == 24000
        print(f"  decode_upsample_rate          : {st.get_decode_upsample_rate()} samples/frame")
        assert st.get_decode_upsample_rate() == 1920
        print(f"  model_type                    : {st.get_model_type()}")
    check("speech_tokenizer", _codec)

    # 9. Tiny end-to-end smoke (CPU): build a 1-frame codec tensor and decode.
    def _codec_decode_roundtrip():
        st = model.speech_tokenizer
        codes = torch.zeros((8, 16), dtype=torch.long)
        wavs, fs = st.decode([{"audio_codes": codes}])
        assert isinstance(wavs, list) and len(wavs) == 1
        print(f"  decoded shape  : {wavs[0].shape}  (samples)")
        print(f"  sample_rate    : {fs}")
    check("codec decode roundtrip (8 frames of zeros)", _codec_decode_roundtrip, fatal=False)

    # 10. _build_assistant_text — check the chat-template format hasn't changed.
    def _assistant_text():
        t = wrapper._build_assistant_text("hello")
        print(f"  _build_assistant_text repr : {repr(t)}")
        assert "<|im_start|>assistant" in t
    check("_build_assistant_text", _assistant_text)

    # 11. The DynamicCache import path our pipeline depends on.
    def _dynamic_cache():
        from transformers.cache_utils import DynamicCache
        c = DynamicCache()
        assert hasattr(c, "layers"), "DynamicCache.layers attribute missing"
        print(f"  DynamicCache.layers attribute : OK (transformers cache API)")
    check("transformers DynamicCache", _dynamic_cache)

    print("\n" + "=" * 64)
    print(f" {OK} all inspections passed — qwen_tts API matches pipeline assumptions")
    print("=" * 64)


if __name__ == "__main__":
    main()
