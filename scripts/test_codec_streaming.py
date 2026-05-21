#!/usr/bin/env python
"""
test_codec_streaming.py — simulates the *real* streaming protocol used by
CodecDecoder (overlap + emit-trailing) and measures quality vs ground truth
for different chunk sizes.

The earlier test_codec_min_chunk.py showed that decoding small chunks IN
ISOLATION has poor quality (1.7-4.5 dB SNR) — but that's not what our code
does. CodecDecoder maintains an overlap of OVERLAP_FRAMES previously-emitted
frames as history, decodes (history + new) together, and emits only the
trailing portion. This test measures THAT.

Question: with OVERLAP_FRAMES=4 fixed, what's the smallest chunk_frames=K
we can use without losing audio quality?

  K=4  baseline (current)
  K=2  chunk = 2 new frames + 4 history = 6 frames per decode, emit 2 frames
  K=1  chunk = 1 new frame + 4 history = 5 frames per decode, emit 1 frame

Smaller K → lower TTFC (we yield the first chunk after fewer frames), but
also more decode calls (less audio per call).
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


OK = "\033[32m✓\033[0m"


def streaming_decode(tokenizer, all_frames, chunk_frames, overlap_frames):
    """Simulate CodecDecoder.decode() with the given chunk and overlap.
    Returns (concatenated_wav float32, total_codec_calls int)."""
    history = []
    out_pieces = []
    n_calls = 0
    SAMPLES_PER_FRAME = 1920

    i = 0
    while i < len(all_frames):
        new_chunk = all_frames[i : i + chunk_frames]
        if len(new_chunk) < 1:
            break

        decode_input = history + new_chunk
        codes = torch.tensor(decode_input, dtype=torch.long)
        wavs, _ = tokenizer.decode([{"audio_codes": codes}])
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)

        # Emit only the trailing |new_chunk| * SAMPLES_PER_FRAME samples
        n_new_samples = min(len(new_chunk) * SAMPLES_PER_FRAME, wav.shape[0])
        new_wav = wav[-n_new_samples:]
        out_pieces.append(new_wav)

        # Rotate history
        history = (history + new_chunk)[-overlap_frames:]
        i += len(new_chunk)
        n_calls += 1

    return np.concatenate(out_pieces) if out_pieces else np.zeros(0, dtype=np.float32), n_calls


def main():
    print("=" * 72)
    print(" Streaming codec quality — chunk + overlap vs ground truth")
    print("=" * 72)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    print("\nLoading model on CPU (~20 s)...")
    wrapper = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        dtype=torch.float32,
        device_map="cpu",
    )
    tokenizer = wrapper.model.speech_tokenizer
    print("  loaded.")

    # Generate a realistic 16-frame test utterance.
    rng = np.random.default_rng(7)
    n_frames = 16
    frames = [
        [int(x) for x in rng.integers(low=10, high=2000, size=16)]
        for _ in range(n_frames)
    ]

    # Ground truth: decode all 16 frames in a single call.
    codes_gt = torch.tensor(frames, dtype=torch.long)
    wavs_gt, _ = tokenizer.decode([{"audio_codes": codes_gt}])
    gt_wav = wavs_gt[0]
    if isinstance(gt_wav, torch.Tensor):
        gt_wav = gt_wav.detach().float().cpu().numpy().reshape(-1)
    else:
        gt_wav = np.asarray(gt_wav).reshape(-1)
    print(f"  ground truth: 1 call decoding {n_frames} frames → {gt_wav.shape[0]} samples")

    # Streaming decodes with overlap=4 and varying chunk sizes.
    print(f"\n  Streaming (overlap=4):")
    print(f"  {'K':>3}  {'calls':>5}  {'samples':>8}  {'max|Δ|':>9}  {'mean|Δ|':>9}  {'SNR':>7}  {'TTFC at 12.5fps':>16}")

    OVERLAP = 4
    results = {}
    for K in [1, 2, 3, 4]:
        wav, n_calls = streaming_decode(tokenizer, frames, chunk_frames=K, overlap_frames=OVERLAP)

        # Compare to ground truth
        L = min(len(wav), len(gt_wav))
        diff = wav[:L] - gt_wav[:L]
        max_abs = float(np.abs(diff).max())
        mean_abs = float(np.abs(diff).mean())
        gt_rms = float(np.sqrt((gt_wav[:L]**2).mean()) + 1e-12)
        err_rms = float(np.sqrt((diff**2).mean()) + 1e-12)
        snr_db = 20 * np.log10(gt_rms / err_rms)

        # TTFC heuristic: you yield the first chunk after K frames are synthesised.
        # At 12.5 frames/sec generation speed, that's K/12.5 sec just for the wait.
        # (Real TTFC also includes prefill + first codec call, ignored here.)
        ttfc_floor_ms = K / 12.5 * 1000

        results[K] = (snr_db, ttfc_floor_ms, n_calls)
        print(f"  {K:>3}  {n_calls:>5}  {len(wav):>8}  {max_abs:>9.4f}  {mean_abs:>9.4f}  {snr_db:>6.1f}dB  {ttfc_floor_ms:>14.0f} ms")

    # Test with longer overlap — does it help small chunks?
    print(f"\n  Streaming with LONGER OVERLAP — does it rescue small-K quality?")
    print(f"  {'K':>3}  {'overlap':>7}  {'calls':>5}  {'SNR':>7}  {'TTFC floor':>11}  {'codec_input':>11}")
    for K, overlap in [(1, 4), (1, 6), (1, 8), (2, 6), (2, 8), (4, 4)]:
        wav, n_calls = streaming_decode(tokenizer, frames, chunk_frames=K, overlap_frames=overlap)
        L = min(len(wav), len(gt_wav))
        diff = wav[:L] - gt_wav[:L]
        gt_rms = float(np.sqrt((gt_wav[:L]**2).mean()) + 1e-12)
        err_rms = float(np.sqrt((diff**2).mean()) + 1e-12)
        snr_db = 20 * np.log10(gt_rms / err_rms)
        ttfc = K / 12.5 * 1000
        per_call = K + overlap
        print(f"  {K:>3}  {overlap:>7}  {n_calls:>5}  {snr_db:>6.1f}dB  {ttfc:>9.0f} ms  {per_call:>4} frames")

    # Pick recommendation
    print("\n" + "=" * 72)
    print(" RECOMMENDATION")
    print("=" * 72)
    # Find smallest K with SNR >= 20 dB (typically inaudible degradation)
    chosen_K = None
    for K in [1, 2, 3, 4]:
        snr, _, _ = results[K]
        if snr >= 20:
            chosen_K = K
            break
    if chosen_K is None:
        # Fall back to best-SNR-within-budget
        chosen_K = min(results, key=lambda k: -results[k][0])
    print(f"\n  Smallest chunk size with SNR ≥ 20 dB: K = {chosen_K}")
    if chosen_K < 4:
        saved = (4 - chosen_K) / 12.5 * 1000
        print(f"  Reducing MIN_CHUNK_FRAMES from 4 → {chosen_K} cuts TTFC floor by ~{saved:.0f} ms")
    else:
        print("  Current default (K=4) is the smallest viable size — keep it.")

    # Always print a summary of all candidates so we can decide
    print("\n  Per-candidate quality summary:")
    for K in [1, 2, 3, 4]:
        snr, ttfc, n_calls = results[K]
        verdict = OK if snr >= 20 else ("marginal" if snr >= 10 else "audible artefacts")
        print(f"    K={K}: {snr:.1f} dB ({verdict})  TTFC floor ≈ {ttfc:.0f} ms")


if __name__ == "__main__":
    main()
