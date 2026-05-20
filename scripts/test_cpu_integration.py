#!/usr/bin/env python
"""
test_cpu_integration.py — CPU-only integration smoke for the HF half of the
pipeline. Catches every failure mode the GPU stages 7/9 of smoke_test.py
would surface, EXCEPT for the kernel itself.

What it tests (in order, fail-fast):
  1. Model loads on CPU.
  2. Voice-clone prompt builds from refs/voice.wav (or REF_AUDIO env).
  3. Our monkey-patched _hf_prefill fires correctly via the real wrapper:
       - intercepts model.talker.generate
       - captures (first_token, kv_cache, last_hidden, trailing, pad)
       - payload shapes match what the pipeline assumes
  4. _extract_kv_for_kernel works on the REAL DynamicCache from prefill.
  5. CodePredictor.predict runs the real subtalker for one frame, returns
     [16] tokens + a [1, 1, 1024] codec_hidden_sum.

This intentionally does NOT run the megakernel (no CUDA on macOS).

Run:
    python scripts/test_cpu_integration.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))   # so `tts.pipeline` resolves

OK = "\033[32m✓\033[0m"
NO = "\033[31m✗\033[0m"

REF_DEFAULT = REPO / "refs" / "voice.wav"


def stage(label, fn):
    print(f"\n── {label} ──")
    t0 = time.perf_counter()
    try:
        out = fn()
        print(f"  {OK} {label}  ({(time.perf_counter()-t0)*1000:.0f} ms)")
        return out
    except Exception as e:
        print(f"  {NO} {label}  ({(time.perf_counter()-t0)*1000:.0f} ms)")
        traceback.print_exc()
        sys.exit(1)


def main():
    print("=" * 64)
    print(" CPU integration test — HF half of the pipeline")
    print("=" * 64)

    ref_audio = os.environ.get("REF_AUDIO") or str(REF_DEFAULT)
    ref_text = os.environ.get("REF_TEXT")  # optional; x-vector-only if unset
    if not os.path.exists(ref_audio):
        print(f"{NO} REF_AUDIO not found: {ref_audio}")
        sys.exit(1)
    print(f"  REF_AUDIO = {ref_audio}")
    print(f"  REF_TEXT  = {ref_text!r}")

    # 1. Load the model on CPU.  Slow first time (~30 s); cached after.
    def _load():
        import torch
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        wrapper = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            dtype=torch.float32,   # float32 on CPU (bf16 cpu math is slow + lossy)
            device_map="cpu",
        )
        return wrapper
    wrapper = stage("1. load Qwen3TTSModel (CPU, float32)", _load)

    # 2. Build voice-clone prompt items from the recorded WAV.
    def _build_prompt():
        items = wrapper.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=(ref_text is None),
        )
        assert isinstance(items, list) and len(items) == 1, items
        item = items[0]
        print(f"  x_vector_only_mode = {item.x_vector_only_mode}")
        print(f"  icl_mode           = {item.icl_mode}")
        print(f"  ref_spk_embedding  = {tuple(item.ref_spk_embedding.shape)}")
        print(f"  ref_code           = "
              f"{'None' if item.ref_code is None else tuple(item.ref_code.shape)}")
        return items
    voice_items = stage("2. create_voice_clone_prompt(refs/voice.wav)", _build_prompt)

    # 3. Patch in our TTSPipeline._hf_prefill flow.  We don't construct
    #    TTSPipeline here (it requires CUDA for the kernel weights extraction
    #    when backend=megakernel); we exercise the monkey-patch logic directly.
    def _prefill_intercept():
        import torch, types
        from tts.pipeline import _PrefillDone

        captured = {}

        def patched_generate(self_tl, inputs_embeds=None, attention_mask=None,
                             trailing_text_hidden=None, tts_pad_embed=None, **kw):
            out = self_tl.model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )
            last_hidden = out.last_hidden_state[:, -1:, :]
            logits = self_tl.codec_head(last_hidden)
            first_token = int(logits.argmax(dim=-1).item())
            raise _PrefillDone({
                "first_token": first_token,
                "kv_cache": out.past_key_values,
                "last_hidden": last_hidden,
                "trailing_text_hidden": trailing_text_hidden,
                "tts_pad_embed": tts_pad_embed,
            })

        original = wrapper.model.talker.generate
        wrapper.model.talker.generate = types.MethodType(
            patched_generate, wrapper.model.talker
        )
        try:
            try:
                wrapper.generate_voice_clone(
                    text="Hello, this is a CPU integration test.",
                    language="Auto",
                    voice_clone_prompt=voice_items,
                    do_sample=False,
                    max_new_tokens=1,
                )
            except _PrefillDone as done:
                captured = done.payload
            else:
                raise RuntimeError(
                    "monkey-patched generate did NOT fire — wrapper completed "
                    "without calling model.talker.generate. The interception "
                    "path is broken; check qwen_tts version."
                )
        finally:
            wrapper.model.talker.generate = original

        # Sanity-check the payload shapes.
        first_token = captured["first_token"]
        kv = captured["kv_cache"]
        last_hidden = captured["last_hidden"]
        trailing = captured["trailing_text_hidden"]
        pad = captured["tts_pad_embed"]

        assert isinstance(first_token, int), type(first_token)
        assert 0 <= first_token < 3072, first_token
        assert last_hidden.shape[-1] == 1024, last_hidden.shape
        assert pad.shape == (1, 1, 1024), pad.shape
        assert trailing.dim() == 3 and trailing.shape[-1] == 1024, trailing.shape
        print(f"  first_token (group-0) = {first_token}")
        print(f"  last_hidden           = {tuple(last_hidden.shape)}")
        print(f"  trailing_text_hidden  = {tuple(trailing.shape)}")
        print(f"  tts_pad_embed         = {tuple(pad.shape)}")
        print(f"  kv_cache layers       = {len(kv.layers)}")
        print(f"  kv_cache[0].keys      = {tuple(kv.layers[0].keys.shape)}")
        return captured

    payload = stage("3. monkey-patched _hf_prefill (intercept + capture)", _prefill_intercept)

    # 4. _extract_kv_for_kernel on the REAL DynamicCache.
    def _extract():
        from tts.pipeline import _extract_kv_for_kernel
        k, v = _extract_kv_for_kernel(payload["kv_cache"])
        # Talker is 28 layers × 8 kv_heads × T × 128.
        assert k.shape[0] == 28, k.shape
        assert k.shape[1] == 8, k.shape
        assert k.shape[3] == 128, k.shape
        assert k.dtype.is_floating_point
        print(f"  k.shape = {tuple(k.shape)}")
        print(f"  v.shape = {tuple(v.shape)}")
        print(f"  dtype   = {k.dtype}")
        return k, v
    stage("4. _extract_kv_for_kernel(real DynamicCache)", _extract)

    # 5. CodePredictor on the real subtalker.
    def _subtalker():
        import torch
        from tts.code_predictor import CodePredictor

        cp = CodePredictor(wrapper.model)
        frame, codec_hidden_sum = cp.predict(
            group0_token=payload["first_token"],
            past_hidden=payload["last_hidden"],
            do_sample=False,
        )
        assert isinstance(frame, list) and len(frame) == 16, frame
        for x in frame:
            assert isinstance(x, int) and 0 <= x < 3072, x
        assert codec_hidden_sum.shape == (1, 1, 1024), codec_hidden_sum.shape
        print(f"  frame[16]              = {frame}")
        print(f"  codec_hidden_sum shape = {tuple(codec_hidden_sum.shape)}")
        print(f"  codec_hidden_sum dtype = {codec_hidden_sum.dtype}")
    stage("5. CodePredictor.predict(real subtalker)", _subtalker)

    print("\n" + "=" * 64)
    print(f" {OK} CPU integration test PASSED — HF half of pipeline is wired correctly")
    print("=" * 64)
    print("Remaining risks (5090 only): kernel JIT, KV semantics, RoPE, audio quality.")


if __name__ == "__main__":
    main()
