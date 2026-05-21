#!/usr/bin/env python
"""
test_codec_min_chunk.py — find the smallest number of frames the
Qwen3-TTS 12 Hz codec will decode.

The official wrapper handles 4+ frames (per our smoke test). If 1-frame
or 2-frame decodes also work — even with edge artifacts — we can cut
TTFC by ~240 ms (the wait for the 4-frame buffer to fill before first
codec call).

For each candidate chunk size, we:
  1. Try to call speech_tokenizer.decode with that many frames
  2. Check for exceptions, NaNs, or zero output
  3. Compare audio quality vs a 4-frame reference (overlap test):
       - Decode 8 frames at once → ground truth wav
       - Decode the same 8 frames in N+8/N chunks → compare
       - Look at the MAX |error| and RMS of the diff
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


OK = "\033[32m✓\033[0m"
NO = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"


def decode_frames(tokenizer, frames):
    """Returns (wav_np float32, error_or_None)."""
    try:
        codes_2d = torch.tensor(frames, dtype=torch.long)
        wavs, fs = tokenizer.decode([{"audio_codes": codes_2d}])
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav, None
    except Exception as e:
        return None, e


def main():
    print("=" * 68)
    print(" Qwen3-TTS codec — smallest viable chunk size")
    print("=" * 68)

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    print("\nLoading model on CPU (float32, ~20 s)...")
    wrapper = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        dtype=torch.float32,
        device_map="cpu",
    )
    tokenizer = wrapper.model.speech_tokenizer
    print("  loaded.")

    # Use deterministic non-trivial frame contents (low integers, all 16 codebooks).
    # Real synthesis produces high integers, but the codec doesn't care about
    # the actual id distribution for this test — we just want to see if the
    # forward pass works.
    def make_frames(n, seed=0):
        rng = np.random.default_rng(seed)
        return [
            [int(x) for x in rng.integers(low=10, high=2000, size=16)]
            for _ in range(n)
        ]

    # Test 1: can the codec handle N frames at all?
    print("\n[Test 1] Decode N frames in one call — does it crash?")
    print(f"  {'N':>3}  {'samples':>9}  {'ratio':>7}  status")
    for n in [1, 2, 3, 4, 5, 8, 12, 16]:
        frames = make_frames(n, seed=1)
        wav, err = decode_frames(tokenizer, frames)
        if err is not None:
            print(f"  {n:>3}    -          -      {NO} {type(err).__name__}: {str(err)[:50]}")
            continue
        samples = wav.shape[0]
        ratio = samples / (n * 1920)  # 1920 samples/frame expected
        has_nan = np.isnan(wav).any()
        all_zero = np.abs(wav).max() < 1e-6
        flags = []
        if has_nan: flags.append("NaN")
        if all_zero: flags.append("ZERO")
        status = OK if not flags else f"{WARN} {', '.join(flags)}"
        print(f"  {n:>3}  {samples:>9}  {ratio:>6.3f}×  {status}")

    # Test 2: quality of N-frame decode vs 8-frame ground truth.
    # Decode 8 frames once → ground truth. Then decode chunks of size N,
    # concatenate. Compare audio.
    print("\n[Test 2] Chunked decode vs 8-frame ground truth")
    print(f"  (decode 8 frames as N chunks of size K, splice, compare to 1×8)")

    gt_frames = make_frames(8, seed=2)
    gt_wav, err = decode_frames(tokenizer, gt_frames)
    if err is not None:
        print(f"  ground truth decode failed: {err}")
        return
    print(f"  ground truth (8 frames in 1 call): {gt_wav.shape[0]} samples")

    print(f"\n  {'chunk_K':>8}  {'max|Δ|':>9}  {'mean|Δ|':>9}  {'~SNR':>7}  notes")

    for K in [1, 2, 3, 4]:
        if 8 % K != 0:
            continue  # only test divisors of 8 for clean comparison
        n_chunks = 8 // K
        pieces = []
        ok = True
        for chunk_idx in range(n_chunks):
            chunk = gt_frames[chunk_idx*K : (chunk_idx+1)*K]
            wav_piece, err = decode_frames(tokenizer, chunk)
            if err is not None:
                print(f"  K={K}: chunk {chunk_idx} failed: {err}")
                ok = False
                break
            pieces.append(wav_piece)
        if not ok:
            continue

        # Sum total samples (may differ from gt due to chunking edge effects)
        spliced = np.concatenate(pieces)
        # Truncate both to min length to compare
        L = min(len(spliced), len(gt_wav))
        diff = spliced[:L] - gt_wav[:L]
        max_abs = float(np.abs(diff).max())
        mean_abs = float(np.abs(diff).mean())
        # SNR proxy: 20*log10(gt_rms / err_rms)
        gt_rms = float(np.sqrt((gt_wav[:L]**2).mean()) + 1e-12)
        err_rms = float(np.sqrt((diff**2).mean()) + 1e-12)
        snr_db = 20 * np.log10(gt_rms / err_rms)
        spliced_len = len(spliced)
        note = ""
        if K == 4:
            note = "(our current default; baseline)"
        elif snr_db > 40:
            note = "essentially identical"
        elif snr_db > 20:
            note = "very minor edge artifacts"
        elif snr_db > 10:
            note = "audible boundary artifacts"
        else:
            note = "noticeable degradation"

        print(f"  K={K:<6} {max_abs:>9.4f}  {mean_abs:>9.4f}  {snr_db:>6.1f}dB  {note}")

    # Test 3: Single-frame decode characteristics
    print("\n[Test 3] What does a single-frame decode look like?")
    f1 = make_frames(1, seed=3)
    wav, err = decode_frames(tokenizer, f1)
    if err is None:
        print(f"  1-frame decode: {wav.shape[0]} samples = {wav.shape[0]/24000*1000:.0f} ms")
        print(f"    expected ~80 ms (12.5 Hz frame rate, 1920 samples/frame)")
        print(f"    abs max:  {np.abs(wav).max():.4f}")
        print(f"    rms:      {np.sqrt((wav**2).mean()):.4f}")
        if wav.shape[0] < 100:
            print(f"  {WARN} output is suspiciously short — codec may be padding/dropping")
    else:
        print(f"  {NO} 1-frame decode failed: {err}")

    print("\n" + "=" * 68)
    print(" CONCLUSION")
    print("=" * 68)
    print("""
  - If 1- and 2-frame decodes succeed AND SNR vs 4-frame is acceptable
    (>20 dB), we can reduce MIN_CHUNK_FRAMES from 4 → 1 and cut TTFC by
    ~240 ms.
  - If they fail or quality is poor at small chunks, we need a different
    streaming strategy (e.g. always feed 4-frame windows but slide by 1).
""")


if __name__ == "__main__":
    main()
