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
    # Probe subtalker / codec attribute presence.
    has_st = any(hasattr(wrapper.model, n) for n in ("subtalker", "code_predictor", "talker_code_predictor"))
    has_codec = hasattr(wrapper.model, "speech_tokenizer")
    print(f"  subtalker present       : {has_st}")
    print(f"  speech_tokenizer present: {has_codec}")
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


@stage("6. Codec decode → PCM file")
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


@stage("7. End-to-end pipeline (HF backend)")
def s7():
    import asyncio
    import numpy as np
    import wave
    from pathlib import Path
    from tts.pipeline import TTSPipeline

    print("  loading pipeline (backend=hf, no voice clone) ...")
    pipe = TTSPipeline(backend="hf")

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


@stage("8. End-to-end pipeline (megakernel backend)")
def s8():
    import asyncio
    import wave
    from pathlib import Path
    from tts.pipeline import TTSPipeline

    print("  loading pipeline (backend=megakernel) ...")
    pipe = TTSPipeline(backend="megakernel")

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
    s6(wrapper)
    s7()
    s8()

    print("\n" + "=" * 60)
    print(f" {OK} all stages passed")
    print("=" * 60)
    print("Listen to ./smoke_test_mega.wav to confirm audio quality.")


if __name__ == "__main__":
    main()
