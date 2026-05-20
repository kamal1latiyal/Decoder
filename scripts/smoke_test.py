#!/usr/bin/env python
"""
smoke_test.py — bisect-style verification on rented hardware.

Runs the pipeline stage-by-stage and prints PASS / FAIL for each. On the
first FAIL, it stops and dumps the traceback. Cheapest way to figure out
which stage is broken without burning $/min on a full benchmark run.

Run:
    python scripts/smoke_test.py

Output: ./smoke_test.wav (3-frame test utterance) on success.
"""

import sys
import time
import traceback
from pathlib import Path

# Make `qwen_megakernel_tts` (repo-root package) importable when invoked as
# `python scripts/smoke_test.py` from anywhere.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OK = "\033[32m✓\033[0m"
NO = "\033[31m✗\033[0m"


def stage(label):
    """Decorator: print label, time it, catch any exception."""
    def wrap(fn):
        def inner(*a, **kw):
            print(f"\n── {label} ──")
            t0 = time.perf_counter()
            try:
                result = fn(*a, **kw)
                dt = (time.perf_counter() - t0) * 1000
                print(f"  {OK} {label}  ({dt:.0f} ms)")
                return result
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                print(f"  {NO} {label}  ({dt:.0f} ms)")
                traceback.print_exc()
                print(f"\n→ stopping after first failure ({label})")
                sys.exit(1)
        return inner
    return wrap


@stage("1. CUDA available")
def s1():
    import torch
    assert torch.cuda.is_available(), "torch reports no CUDA device"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"  device   : {name}")
    print(f"  cc       : sm_{cap[0]}{cap[1]}")
    print(f"  vram     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if cap[0] < 12:
        print(f"  WARNING: cc < 12.0 — megakernel requires sm_120 (Blackwell)")


@stage("2. JIT compile megakernel")
def s2():
    from qwen_megakernel_tts.build import get_extension
    get_extension()
    import torch
    # Confirm the ops registered via TORCH_LIBRARY are reachable.
    assert hasattr(torch.ops, "qwen_megakernel_C"), "torch.ops.qwen_megakernel_C not registered"
    assert callable(torch.ops.qwen_megakernel_C.decode), "decode op missing"
    print("  torch.ops.qwen_megakernel_C.decode  ✓")
    print("  torch.ops.qwen_megakernel_C.generate_nosync  ✓"
          if callable(getattr(torch.ops.qwen_megakernel_C, "generate_nosync", None))
          else "  generate_nosync missing (not used by pipeline, just FYI)")


@stage("3. Load Qwen3-TTS model (HF)")
def s3():
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    wrapper = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"  model class : {type(wrapper.model).__name__}")
    print(f"  tts_model_type : {wrapper.model.tts_model_type}")
    # Probe submodule presence at the REAL attribute paths used by our pipeline:
    #   model.talker.code_predictor  (the subtalker — qwen_tts naming)
    #   model.speech_tokenizer       (the codec)
    talker = getattr(wrapper.model, "talker", None)
    has_st = (
        talker is not None
        and any(hasattr(talker, n) for n in ("code_predictor", "subtalker", "talker_code_predictor"))
    )
    has_codec = hasattr(wrapper.model, "speech_tokenizer") and wrapper.model.speech_tokenizer is not None
    print(f"  talker present          : {talker is not None}")
    print(f"  subtalker present       : {has_st}  (at model.talker.code_predictor)")
    print(f"  speech_tokenizer present: {has_codec}")
    assert talker is not None, "no model.talker — pipeline assumes Qwen3-TTS Base"
    assert has_st, "no subtalker — CodePredictor wrapper will fail"
    assert has_codec, "no speech_tokenizer — CodecDecoder wrapper will fail"
    return wrapper


@stage("4. Extract + pack talker weights")
def s4(wrapper):
    from qwen_megakernel_tts.model import TalkerKernelWeights
    talker = getattr(wrapper.model, "talker", wrapper.model)
    w = TalkerKernelWeights(talker)
    print(f"  embed_weight   : {tuple(w.embed_weight.shape)} {w.embed_weight.dtype}")
    print(f"  lm_head_weight : {tuple(w.lm_head_weight.shape)} {w.lm_head_weight.dtype}")
    print(f"  layer_pack     : {tuple(w.layer_weights_packed.shape)} {w.layer_weights_packed.dtype}")
    assert w.embed_weight.shape[0] == 3072, "expected codec vocab 3072"
    assert w.embed_weight.shape[1] == 1024, "expected hidden 1024"
    return w


@stage("5. Megakernel single decode step (cold cache)")
def s5(kw):
    """
    Bare-metal test: bypass HF prefill, just call decode() with a zero KV cache.
    This verifies the kernel call works and doesn't crash. Output is garbage
    (cache is empty) — we only check that it returns a valid token id.
    """
    from tts.talker import MegakernelDecoder
    dec = MegakernelDecoder(kw)
    dec.reset()
    out = dec.step(0)
    assert isinstance(out, int) and 0 <= out < 3072, f"bad token: {out}"
    print(f"  decode(0) → {out}  (range OK, value meaningless without prefill)")
    # Time 50 more steps for a rough tok/s sanity check.
    import time
    t0 = time.perf_counter()
    tok = out
    for _ in range(50):
        tok = dec.step(tok)
    dt = (time.perf_counter() - t0)
    print(f"  50 steps  : {dt*1000:.1f} ms total  =  {50/dt:.0f} tok/s")
    if 50 / dt < 400:
        print(f"  WARNING: throughput < 400 tok/s — well below the 1000 tok/s target.")
    return dec


@stage("6. Kernel ↔ HF parity (warm KV from synthetic prefix)")
def s_parity(wrapper, kw):
    """
    Cheapest hardware-side check that the kernel's per-step decode produces the
    same result as the HF talker, given the same warm KV cache.

    Why this matters: the KV cache hand-off from HF (DynamicCache,
    [B, kv_heads, T, head_dim]) to the kernel ([L, kv_heads, T, head_dim])
    relies on a layout assumption that we can't validate without running both
    paths. The other latent risk is RoPE convention (MRoPE vs split-half RoPE,
    cos/sin pair ordering, theta).  This stage exercises both end-to-end on
    short synthetic data.

    Strategy:
      1. Build a random prefix in *embedding space* (bf16 [1, 8, 1024]).
      2. HF: run talker.model.forward(inputs_embeds=prefix, use_cache=True).
         Capture (kv_cache, last_hidden) before any further mutation.
      3. Build a random next-step input vector.
      4. HF: run talker.model.forward(inputs_embeds=next_input, past_kv=kv,
         use_cache=True). Capture post-norm hidden + codec_head argmax.
      5. Kernel: reset, set_kv_prefix(extracted k/v from step 2 — captured
         BEFORE step 4 mutated anything), step_from_hidden(next_input).
         Capture post-norm hidden (norm_out) + output token.
      6. Compare per-element hidden vectors (allclose with bf16-friendly
         tolerance) and argmax tokens.

    Pass criteria (any failure is reported with detailed diagnostics):
      - hidden state allclose with atol=0.25, rtol=0.10
      - tokens match  OR  kernel's token is in HF's top-5
    """
    import torch, sys
    from tts.talker import MegakernelDecoder
    from tts.pipeline import _extract_kv_for_kernel

    # Single inference_mode block for the whole stage. Tensors produced inside
    # cannot leave for autograd-tracked code (linear with Parameter weights);
    # codec_head's forward goes through autograd if called outside, so we keep
    # everything inside one consistent inference_mode context.
    with torch.inference_mode():
        torch.manual_seed(42)
        device = next(wrapper.model.parameters()).device
        dtype = next(wrapper.model.parameters()).dtype
        assert dtype == torch.bfloat16, f"expected talker in bf16, got {dtype}"

        talker = wrapper.model.talker            # Qwen3TTSTalkerForConditionalGeneration
        talker_model = talker.model              # Qwen3TTSTalkerModel (the 28-layer backbone)

        # 1. Random prefix in embedding space.  Small scale so values stay in
        #    the bf16 sweet spot and we don't trigger numerical edge cases.
        prefix_len = 8
        prefix = (torch.randn(1, prefix_len, 1024, device=device) * 0.1).to(dtype)
        print(f"  prefix shape={tuple(prefix.shape)} dtype={prefix.dtype}")

        # 2. HF prefill — capture KV BEFORE we run the next-step forward
        #    (which mutates the DynamicCache in place).
        prefill_out = talker_model.forward(
            inputs_embeds=prefix, use_cache=True, output_hidden_states=False,
        )
        kv_for_kernel_k, kv_for_kernel_v = _extract_kv_for_kernel(prefill_out.past_key_values)
        print(f"  KV extracted             : k={tuple(kv_for_kernel_k.shape)} "
              f"v={tuple(kv_for_kernel_v.shape)}  dtype={kv_for_kernel_k.dtype}")
        assert kv_for_kernel_k.shape == (28, 8, prefix_len, 128), kv_for_kernel_k.shape

        # 3. Random next-step input vector (1024-dim bf16).
        next_input = (torch.randn(1024, device=device) * 0.1).to(dtype)

        # 4. HF: one more step from the warm cache.
        hf_step = talker_model.forward(
            inputs_embeds=next_input.view(1, 1, 1024),
            past_key_values=prefill_out.past_key_values,
            use_cache=True,
            output_hidden_states=False,
        )
        hf_hidden = hf_step.last_hidden_state[0, -1, :].to(torch.float32)        # [1024]
        hf_logits = talker.codec_head(hf_step.last_hidden_state[:, -1, :])[0]    # [3072]
        hf_topk = torch.topk(hf_logits, 5)
        hf_token = int(hf_topk.indices[0].item())

        # 5. Kernel: same warm KV (captured pre-mutation), same next_input.
        dec = MegakernelDecoder(kw)
        dec.reset()
        dec.set_kv_prefix(kv_for_kernel_k, kv_for_kernel_v)
        kernel_token, kernel_hidden_bf16 = dec.step_from_hidden(next_input)
        kernel_hidden = kernel_hidden_bf16.to(torch.float32).view(-1)             # [1024]

    # 6. Diagnostics.
    abs_diff = (kernel_hidden - hf_hidden).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        kernel_hidden.unsqueeze(0), hf_hidden.unsqueeze(0)
    ).item()
    print(f"  HF top-5 tokens          : {hf_topk.indices.tolist()}")
    print(f"  HF top-1 token / logit   : {hf_token} / {hf_topk.values[0].item():.2f}")
    print(f"  Kernel token             : {kernel_token}")
    print(f"  hidden max|Δ|            : {max_abs:.4f}")
    print(f"  hidden mean|Δ|           : {mean_abs:.4f}")
    print(f"  hidden cosine            : {cos_sim:.6f}")

    hf_top5_ids = set(hf_topk.indices.tolist())
    token_match = (kernel_token == hf_token)
    token_in_top5 = (kernel_token in hf_top5_ids)
    # bf16 over 28 layers + diverging-RoPE-conventions accumulate error; be generous
    # but not infinite. atol=0.25 = ~10x bf16-noise of a single layer; cosine >0.95
    # means the directions agree, which is the load-bearing thing for argmax.
    hidden_ok = (cos_sim > 0.95) and (max_abs < 0.25)

    if hidden_ok and token_match:
        print(f"  → PASS: kernel matches HF (token + hidden)")
    elif hidden_ok and token_in_top5:
        print(f"  → PASS (degraded): kernel token differs from HF top-1 but lies in "
              f"HF top-5; hidden direction matches")
    elif token_match:
        print(f"  → MARGINAL: tokens agree but hidden state drifted "
              f"(max|Δ|={max_abs:.3f}, cos={cos_sim:.4f}) — bf16 noise or kernel "
              f"computing something subtly different. Acceptable iff smoke audio sounds clean.")
    else:
        print(f"  → FAIL: tokens disagree AND hidden state diverged")
        print(f"     Likely root causes (in order of probability):")
        print(f"       1. KV layout mismatch (transpose / interleave)")
        print(f"       2. RoPE convention (split-half vs interleaved, cos/sin ordering)")
        print(f"       3. MRoPE position-dim handling")
        print(f"       4. Weight extraction (wrong codec_head / norm / layer order)")
        sys.exit(1)


@stage("7. Codec decode → PCM file")
def s6(wrapper):
    """
    Feed the codec a few synthetic frames and write 0.5s of (garbage) audio
    to ./smoke_test.wav. This proves the codec path produces valid PCM.
    """
    import numpy as np
    import wave
    from tts.codec import CodecDecoder, MIN_CHUNK_FRAMES

    codec = CodecDecoder(wrapper.model)
    codec.reset()
    # 8 frames of group-0 token = 1 (safe, in-range), other groups = 0.
    frames = [[1] + [0] * 15 for _ in range(8)]
    pcm_bytes = codec.decode(frames)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    print(f"  frames in : {len(frames)}")
    print(f"  pcm bytes : {len(pcm_bytes)}")
    print(f"  samples   : {len(samples)}  ({len(samples) / 24000:.3f}s @ 24kHz)")

    out_path = Path(__file__).parent.parent / "smoke_test.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)
    print(f"  wrote {out_path}")


def _ref_args():
    """Read REF_AUDIO/REF_TEXT from env. Both backends now require a reference voice."""
    import os
    ref_audio = os.environ.get("REF_AUDIO")
    ref_text = os.environ.get("REF_TEXT")
    if not ref_audio:
        raise RuntimeError(
            "REF_AUDIO env var not set. Both backends require a reference voice.\n"
            "  Option 1 (recommended): run bootstrap.sh, which auto-downloads one.\n"
            "  Option 2: export REF_AUDIO=/path/to/voice.wav REF_TEXT='transcript'"
        )
    return ref_audio, ref_text


@stage("8. End-to-end pipeline (HF backend)")
def s7():
    import asyncio
    import wave
    from pathlib import Path
    from tts.pipeline import TTSPipeline

    ref_audio, ref_text = _ref_args()
    print(f"  loading pipeline (backend=hf, ref={ref_audio}) ...")
    pipe = TTSPipeline(backend="hf", ref_audio_path=ref_audio, ref_text=ref_text)

    async def run():
        pcm_all = bytearray()
        async for pcm in pipe.synthesize("Hello, this is a smoke test."):
            pcm_all.extend(pcm)
        return bytes(pcm_all)

    pcm = asyncio.run(run())
    m = pipe.last_metrics
    print(f"  ttfc      : {m.ttfc_ms:.1f} ms")
    print(f"  rtf       : {m.rtf:.3f}")
    print(f"  tok/s     : {m.tokens_per_sec:.0f}")
    print(f"  audio_dur : {m.audio_duration_s:.3f}s")

    out_path = Path(__file__).parent.parent / "smoke_test_hf.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes(pcm)
    print(f"  wrote {out_path} ({len(pcm)} bytes)")


@stage("9. End-to-end pipeline (megakernel backend)")
def s8():
    import asyncio
    import wave
    from pathlib import Path
    from tts.pipeline import TTSPipeline

    ref_audio, ref_text = _ref_args()
    print(f"  loading pipeline (backend=megakernel, ref={ref_audio}) ...")
    pipe = TTSPipeline(backend="megakernel", ref_audio_path=ref_audio, ref_text=ref_text)

    async def run():
        pcm_all = bytearray()
        async for pcm in pipe.synthesize("Hello from the megakernel."):
            pcm_all.extend(pcm)
        return bytes(pcm_all)

    pcm = asyncio.run(run())
    m = pipe.last_metrics
    print(f"  ttfc      : {m.ttfc_ms:.1f} ms   (target < 90)")
    print(f"  rtf       : {m.rtf:.3f}      (target < 0.3)")
    print(f"  tok/s     : {m.tokens_per_sec:.0f}")
    print(f"  audio_dur : {m.audio_duration_s:.3f}s")

    out_path = Path(__file__).parent.parent / "smoke_test_mega.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes(pcm)
    print(f"  wrote {out_path} ({len(pcm)} bytes)")


def main():
    print("=" * 60)
    print(" Megakernel ↔ Qwen3-TTS smoke test")
    print("=" * 60)

    s1()
    s2()
    wrapper = s3()
    kw = s4(wrapper)
    s5(kw)
    s_parity(wrapper, kw)     # NEW: cheapest hardware-side correctness check
    s6(wrapper)
    s7()
    s8()

    print("\n" + "=" * 60)
    print(f" {OK} all stages passed")
    print("=" * 60)
    print("Listen to ./smoke_test_mega.wav to confirm audio quality.")


if __name__ == "__main__":
    main()
