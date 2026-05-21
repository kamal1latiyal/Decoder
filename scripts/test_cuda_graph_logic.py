#!/usr/bin/env python
"""
test_cuda_graph_logic.py — CPU verification that the manual 15-step decode
loop in CUDAGraphedCodePredictor produces the SAME tokens as HF's
generate(do_sample=False).

Runs on Mac (or any CPU box) — no GPU required. Doesn't exercise the actual
CUDA graph capture (that's GPU-only), but verifies the *numerical math* of
the manual loop. If outputs match here, the GPU run will work modulo any
graph-capture quirks (which are caught by the auto-fallback in pipeline.py).

What's tested:
  1. Model loads on CPU (float32).
  2. Reference: HF CodePredictor.predict(group0_token, past_hidden,
     do_sample=False) → frame_a, codec_hidden_sum_a
  3. New: CUDAGraphedCodePredictor.predict(...)  (CPU test mode — runs the
     manual loop directly each call instead of replaying a graph)
     → frame_b, codec_hidden_sum_b
  4. Asserts:
       - frame_a == frame_b      (greedy decode is deterministic; tokens must match)
       - codec_hidden_sum_a ≈ codec_hidden_sum_b (float-equal within tolerance)

Run:
    python scripts/test_cuda_graph_logic.py
"""

import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


OK = "\033[32m✓\033[0m"
NO = "\033[31m✗\033[0m"


def main():
    print("=" * 68)
    print(" CUDAGraphedCodePredictor — CPU math-correctness test")
    print("=" * 68)

    # ── 1. Load model on CPU ──────────────────────────────────────────
    print("\n[1] Loading Qwen3TTSModel on CPU (float32, ~30 s)...")
    t0 = time.perf_counter()
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    wrapper = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        dtype=torch.float32,
        device_map="cpu",
    )
    model = wrapper.model
    print(f"    loaded in {time.perf_counter() - t0:.1f} s")

    # ── 2. Construct both predictors ──────────────────────────────────
    from tts.code_predictor import CodePredictor, CUDAGraphedCodePredictor

    print("\n[2] Constructing predictors...")
    ref = CodePredictor(model)
    print(f"    {OK} CodePredictor (HF reference)")
    cgp = CUDAGraphedCodePredictor(model)
    print(f"    {OK} CUDAGraphedCodePredictor "
          f"(cpu_test_mode={cgp._cpu_test_mode}, using_compile={cgp._using_compile})")

    # ── 3. Prepare a representative input ─────────────────────────────
    # Real past_hidden would come from MegakernelDecoder.step_from_hidden's
    # 2nd return; for math equivalence we just need a deterministic vector.
    torch.manual_seed(42)
    H = model.talker.code_predictor.config.hidden_size
    past_hidden = (torch.randn(1, 1, H) * 0.1).to(torch.float32)
    group0_token = 1234  # any in-range codec id (vocab is 3072)

    print(f"\n[3] Inputs prepared:")
    print(f"    group0_token = {group0_token}")
    print(f"    past_hidden  = {tuple(past_hidden.shape)} {past_hidden.dtype}")

    # ── 4. Run reference (HF, do_sample=False) ───────────────────────
    print("\n[4] Running HF reference (do_sample=False)...")
    t0 = time.perf_counter()
    frame_a, sum_a = ref.predict(
        group0_token, past_hidden,
        do_sample=False, top_k=1, top_p=1.0, temperature=1.0,
    )
    t_ref = time.perf_counter() - t0
    print(f"    {OK} frame   = {frame_a}  ({t_ref*1000:.0f} ms)")
    print(f"      sum.shape  = {tuple(sum_a.shape)}")
    print(f"      sum.norm   = {sum_a.float().norm().item():.4f}")

    # ── 5. Run new path (manual loop) ─────────────────────────────────
    print("\n[5] Running CUDAGraphedCodePredictor (manual loop, CPU mode)...")
    t0 = time.perf_counter()
    frame_b, sum_b = cgp.predict(group0_token, past_hidden)
    t_new = time.perf_counter() - t0
    print(f"    {OK} frame   = {frame_b}  ({t_new*1000:.0f} ms)")
    print(f"      sum.shape  = {tuple(sum_b.shape)}")
    print(f"      sum.norm   = {sum_b.float().norm().item():.4f}")

    # ── 6. Assertions ─────────────────────────────────────────────────
    print("\n[6] Verifying equivalence...")

    if frame_a == frame_b:
        print(f"    {OK} tokens match exactly")
    else:
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(frame_a, frame_b)) if a != b]
        print(f"    {NO} tokens DIFFER at {len(diffs)} positions:")
        for i, a, b in diffs[:5]:
            print(f"         position {i}: ref={a}  new={b}")
        sys.exit(1)

    # Codec hidden sum should match within float32 noise.
    sum_a_f = sum_a.float().reshape(-1)
    sum_b_f = sum_b.float().reshape(-1)
    abs_diff = (sum_a_f - sum_b_f).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        sum_a_f.unsqueeze(0), sum_b_f.unsqueeze(0)
    ).item()
    print(f"    codec_hidden_sum max|Δ| = {abs_diff.max().item():.6e}")
    print(f"    codec_hidden_sum cos    = {cos_sim:.8f}")

    if torch.allclose(sum_a_f, sum_b_f, atol=1e-3, rtol=1e-3) or cos_sim > 0.9999:
        print(f"    {OK} codec_hidden_sum match (within float noise)")
    else:
        print(f"    {NO} codec_hidden_sum diverged beyond tolerance")
        sys.exit(1)

    print("\n" + "=" * 68)
    print(f" {OK} ALL CHECKS PASS — manual decode loop is numerically equivalent")
    print(f"   to HF generate(do_sample=False).")
    print(f"   GPU run should produce identical output, faster.")
    print("=" * 68)
    print(f"\n   Per-call cost on this CPU (float32, unoptimised):")
    print(f"     HF reference : {t_ref*1000:>7.0f} ms")
    print(f"     Manual loop  : {t_new*1000:>7.0f} ms")
    print(f"   (On GPU + bf16 + CUDA graph: HF ~40 ms, manual+graph ~5-10 ms.)")


if __name__ == "__main__":
    main()
