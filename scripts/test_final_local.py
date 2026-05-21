#!/usr/bin/env python
"""
test_final_local.py — comprehensive pre-GPU regression suite.

Runs all CPU-testable paths in one go to catch any regression from the 4
perf optimizations (uvloop, codec pre-alloc, configurable chunks, CUDA
streams via submit/collect).

What this exercises (each section fails loudly + stops on first error):

  A. uvloop is the active asyncio policy
  B. Imports + class construction without GPU
  C. CodecDecoder under default config — produces same bytes as decode()
     via submit()+collect()
  D. CodecDecoder under aggressive config (chunk=1, overlap=8) — still
     produces sensible audio
  E. CUDAGraphedCodePredictor (CPU test mode) — tokens match HF reference
     bit-exact (the rejection-fix regression)
  F. HF prefill monkey-patch works against the real wrapper + voice.wav
  G. _extract_kv_for_kernel handles the real DynamicCache from prefill

If everything passes, you're as safe as you can be on Mac to rent the GPU.
"""

import os
import sys
import time
from pathlib import Path

OK = "\033[32m✓\033[0m"
NO = "\033[31m✗\033[0m"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

VOICE_WAV = REPO / "refs" / "voice.wav"


def section(label):
    """Print a section header + capture exceptions cleanly."""
    def deco(fn):
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
                import traceback
                traceback.print_exc()
                sys.exit(1)
        return inner
    return deco


@section("A. uvloop active")
def section_a():
    import asyncio
    import uvloop
    uvloop.install()
    loop = asyncio.new_event_loop()
    assert "uvloop" in type(loop).__module__, f"loop is {type(loop)}, expected uvloop"
    print(f"  event loop: {type(loop).__module__}.{type(loop).__name__}")
    loop.close()


@section("B. Imports + module-level construction")
def section_b():
    # All the modules that change in this work should import cleanly
    from tts import codec, pipeline, code_predictor, talker
    from qwen_megakernel_tts import model as qmt_model
    from pipecat_integration import voice_loop
    from server import app as server_app

    # Confirm key symbols exist after refactor
    assert hasattr(codec, "DEFAULT_CHUNK_FRAMES"), "DEFAULT_CHUNK_FRAMES missing"
    assert hasattr(codec, "DEFAULT_OVERLAP_FRAMES"), "DEFAULT_OVERLAP_FRAMES missing"
    assert hasattr(codec, "MIN_CHUNK_FRAMES"), "MIN_CHUNK_FRAMES missing"
    assert hasattr(codec, "_DecodeHandle"), "_DecodeHandle (submit/collect handle) missing"
    assert hasattr(code_predictor, "CUDAGraphedCodePredictor"), "CUDAGraphedCodePredictor missing"

    print(f"  codec.DEFAULT_CHUNK_FRAMES   = {codec.DEFAULT_CHUNK_FRAMES}")
    print(f"  codec.DEFAULT_OVERLAP_FRAMES = {codec.DEFAULT_OVERLAP_FRAMES}")
    print(f"  codec.MIN_CHUNK_FRAMES (hard floor) = {codec.MIN_CHUNK_FRAMES}")


_wrapper = None


def _load_wrapper():
    """Singleton model load — ~20s. Used by sections C-G."""
    global _wrapper
    if _wrapper is not None:
        return _wrapper
    print("  Loading model on CPU (one-time, ~20s)...")
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    _wrapper = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        dtype=torch.float32,
        device_map="cpu",
    )
    return _wrapper


@section("C. CodecDecoder default config: decode() ≡ submit()+collect()")
def section_c():
    import torch
    from tts.codec import CodecDecoder

    wrapper = _load_wrapper()
    cd = CodecDecoder(wrapper.model)
    print(f"  chunk_frames={cd.min_chunk_frames}, overlap_frames={cd.overlap_frames}")
    print(f"  device={cd.device}, codec_stream={cd._codec_stream}")
    assert cd._codec_stream is None, "CPU should have no codec stream"

    test_frames_1 = [[100 + i, 200 + i, 300 + i] + [0]*13 for i in range(4)]
    test_frames_2 = [[400 + i, 500 + i, 600 + i] + [0]*13 for i in range(4)]

    # Path 1: synchronous decode()
    cd.reset()
    pcm_sync_1 = cd.decode(test_frames_1)
    pcm_sync_2 = cd.decode(test_frames_2)

    # Path 2: submit() + collect()
    cd.reset()
    h1 = cd.submit(test_frames_1)
    pcm_async_1 = cd.collect(h1)
    h2 = cd.submit(test_frames_2)
    pcm_async_2 = cd.collect(h2)

    assert pcm_sync_1 == pcm_async_1, "first chunk: sync != async"
    assert pcm_sync_2 == pcm_async_2, "second chunk: sync != async"
    print(f"  chunk-1: {len(pcm_sync_1)} bytes  (sync == async ✓)")
    print(f"  chunk-2: {len(pcm_sync_2)} bytes  (sync == async ✓)")

    # collect() is idempotent
    cd.reset()
    h = cd.submit(test_frames_1)
    a = cd.collect(h)
    b = cd.collect(h)
    assert a == b
    print(f"  collect() idempotent: ✓")


@section("D. CodecDecoder aggressive config (chunk=1, overlap=8)")
def section_d():
    from tts.codec import CodecDecoder

    wrapper = _load_wrapper()
    cd = CodecDecoder(wrapper.model, chunk_frames=1, overlap_frames=8)
    assert cd.min_chunk_frames == 1
    assert cd.overlap_frames == 8

    # Stream 8 frames one at a time
    frames = [[100 + i*7, 200 + i*7, 300 + i*7] + [0]*13 for i in range(8)]
    cd.reset()
    pcm_total = b""
    for i, fr in enumerate(frames):
        pcm = cd.decode([fr])
        # Each call should emit 1 frame = 1920 samples = 3840 bytes
        assert len(pcm) == 1920 * 2, f"call {i}: got {len(pcm)}, expected 3840"
        pcm_total += pcm

    total_samples = len(pcm_total) // 2
    print(f"  8 calls × 1 frame each → {total_samples} samples = "
          f"{total_samples / 24000 * 1000:.0f} ms of audio")
    assert total_samples == 8 * 1920


@section("E. CUDAGraphedCodePredictor tokens match HF reference (bit-exact)")
def section_e():
    import torch
    from tts.code_predictor import CodePredictor, CUDAGraphedCodePredictor

    wrapper = _load_wrapper()
    ref = CodePredictor(wrapper.model)
    cgp = CUDAGraphedCodePredictor(wrapper.model)
    print(f"  cpu_test_mode={cgp._cpu_test_mode}, graph={cgp._graph}")

    torch.manual_seed(42)
    H = wrapper.model.talker.code_predictor.config.hidden_size
    past_hidden = (torch.randn(1, 1, H) * 0.1).to(torch.float32)
    g0 = 1234

    frame_a, sum_a = ref.predict(g0, past_hidden,
                                  do_sample=False, top_k=1, top_p=1.0, temperature=1.0)
    frame_b, sum_b = cgp.predict(g0, past_hidden)

    assert frame_a == frame_b, f"tokens differ:\n  ref={frame_a}\n  cgp={frame_b}"
    diff = (sum_a.float() - sum_b.float()).abs().max().item()
    assert diff < 1e-3, f"codec_hidden_sum max|Δ| = {diff}"
    print(f"  frame bit-exact (16/16 match): {frame_a}")
    print(f"  codec_hidden_sum max|Δ| = {diff:.2e}")


@section("F + G. HF prefill monkey-patch + KV extraction")
def section_fg():
    if not VOICE_WAV.exists():
        print(f"  SKIP — needs {VOICE_WAV} (run scripts/bootstrap.sh's default first, or scp from box)")
        return
    import torch, types
    wrapper = _load_wrapper()
    from tts.pipeline import _PrefillDone, _extract_kv_for_kernel

    items = wrapper.create_voice_clone_prompt(
        ref_audio=str(VOICE_WAV), ref_text=None, x_vector_only_mode=True,
    )
    captured = {}

    def patched_generate(self_tl, inputs_embeds=None, attention_mask=None,
                         trailing_text_hidden=None, tts_pad_embed=None, **kw):
        out = self_tl.model.forward(inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask, use_cache=True)
        last_hidden = out.last_hidden_state[:, -1:, :]
        first_token = int(self_tl.codec_head(last_hidden).argmax(dim=-1).item())
        raise _PrefillDone({
            "first_token": first_token,
            "kv_cache": out.past_key_values,
            "last_hidden": last_hidden,
            "trailing_text_hidden": trailing_text_hidden,
            "tts_pad_embed": tts_pad_embed,
        })

    original = wrapper.model.talker.generate
    wrapper.model.talker.generate = types.MethodType(patched_generate, wrapper.model.talker)
    try:
        try:
            wrapper.generate_voice_clone(
                text="Hello, this is a test.", language="Auto",
                voice_clone_prompt=items, do_sample=False, max_new_tokens=1,
            )
        except _PrefillDone as e:
            captured = e.payload
        else:
            raise RuntimeError("monkey-patched generate did not fire")
    finally:
        wrapper.model.talker.generate = original

    k, v = _extract_kv_for_kernel(captured["kv_cache"])
    print(f"  first_token        = {captured['first_token']}")
    print(f"  last_hidden        = {tuple(captured['last_hidden'].shape)}")
    print(f"  KV (kernel layout) = k={tuple(k.shape)}  v={tuple(v.shape)}")
    assert k.shape[0] == 28 and k.shape[1] == 8 and k.shape[3] == 128


def main():
    print("=" * 68)
    print(" Final pre-GPU local regression suite")
    print("=" * 68)
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_fg()
    print("\n" + "=" * 68)
    print(f" {OK} ALL CHECKS PASS — safe to rent the GPU")
    print("=" * 68)


if __name__ == "__main__":
    main()
