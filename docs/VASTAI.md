# Renting an RTX 5090 on Vast.ai

Practical checklist for spinning up a Blackwell box that can JIT-build the
megakernel and run this pipeline end-to-end.

---

## 1. Pre-flight (local, 5 minutes)

- [ ] Push this repo to GitHub (private is fine). See the bottom of the main
      `README.md` for the one-shot push command.
- [ ] Make sure you have an **SSH key** locally and the public half added to
      Vast.ai → *Account → SSH Keys*. (You'll need to `ssh-keygen` if not.)
- [ ] Pick a payment cap. $10 covers ~10 hours at typical 5090 prices and is
      far more than you need for a single run.

---

## 2. Filters to set on the Vast.ai search page

| Filter            | Value                  | Why |
|-------------------|------------------------|-----|
| **GPU**           | RTX 5090               | Required (sm_120). |
| **CUDA Version**  | ≥ 12.8                 | Blackwell support landed in 12.8. |
| **Driver Version**| ≥ 575                  | Same — older drivers don't expose sm_120. |
| **Disk Space**    | ≥ 25 GB                | Models (3 GB) + torch (5 GB) + kernel build + headroom. |
| **RAM**           | ≥ 16 GB                | Plenty for the loader; nothing here is CPU-bound. |
| **Inet Down**     | ≥ 300 Mbit/s           | The HF model download is ~3 GB; saves 1–2 min. |
| **Reliability**   | ≥ 95%                  | Avoid flaky hosts. |
| **DLPerf**        | (sort desc.)           | Use as a tiebreaker between equal-price boxes. |
| **Verified**      | ON                     | Host has passed Vast.ai's verification. |
| **Price**         | $/hr cap to taste      | RTX 5090 typically $0.50–$1.50/hr on-demand. |

Also useful: in the **Image** field pick a CUDA 12.8 image **with `nvcc`**.
Recommended (one of):

- `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`  ← has nvcc, ready to go
- `nvcr.io/nvidia/pytorch:25.01-py3`             ← NGC, also has nvcc
- Vast.ai's "PyTorch (cuDNN Devel)" template

⚠️ **Do NOT pick a `-runtime` image** — those ship without `nvcc` and the
JIT kernel build will fail. If you accidentally pick one, you can
`apt install cuda-toolkit-12-8` but it adds 5 min and a few GB.

---

## 3. Per-listing checks (before you click "Rent")

For each candidate machine, hover/click for details and verify:

- [ ] **CUDA 12.8+** (not 12.4, not 12.6 — the kernel build will fail).
- [ ] **Driver 575+** (`nvidia-smi` row in the listing).
- [ ] **No "Interruptible" / spot** unless you're OK losing the box mid-run.
      For a 2–4 hour benchmarking session, on-demand is worth it.
- [ ] **Single-tenant** preferred (some hosts split a node across renters).
- [ ] **Location**: pick one geographically close to you for SSH responsiveness.
      A 200 ms RTT makes editing painful. Aim for < 100 ms.

---

## 4. Spinning it up (3 minutes)

1. Click **Rent** on a candidate listing.
2. Vast.ai gives you an SSH command like
   `ssh -p 12345 root@ssh4.vast.ai`. Copy it.
3. Add `-A` and your key:
   ```bash
   ssh -A -i ~/.ssh/id_ed25519 -p 12345 root@ssh4.vast.ai
   ```
4. Once you're in:
   ```bash
   nvidia-smi | head -3       # confirm RTX 5090 + driver 575+
   nvcc --version             # confirm CUDA 12.8+
   ```
   If either fails, stop the instance and pick a different listing —
   don't waste time fighting it.

---

## 5. Bootstrapping the repo (one command, ~5 min)

```bash
# Replace <YOUR-USER> + <REPO> with your GitHub fork.
git clone https://github.com/<YOUR-USER>/<REPO>.git Decoder
cd Decoder
bash scripts/bootstrap.sh
```

`bootstrap.sh` does, in order:
1. `scripts/install.sh` — clone the kernel repo, build the extension, install deps
2. `scripts/download_models.py` — pull both HF models
3. `scripts/smoke_test.py` — bisect-style: kernel build → weight load → 10 decode steps → codec → PCM file
4. Start the TTS server in the background
5. Run all three benchmarks (TTFC, RTF, E2E), saving results to `benchmarks/results/`

Total: ~10 minutes if nothing breaks. ~30 min if the KV-layout hand-off
needs tweaking (the one piece of `tts/pipeline.py:_hf_prefill` that's coded
against documented HF shapes but not yet verified on hardware).

---

## 6. When you're done

```bash
# Pull benchmark results back home
scp -P 12345 root@ssh4.vast.ai:~/Decoder/benchmarks/results/*.json ./

# Stop the instance from the Vast.ai web UI (otherwise it keeps billing)
```

Don't just SSH-disconnect — the box keeps running and charging until you hit
**Destroy** in the web UI.

---

## 7. If something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvcc: command not found` | `-runtime` image | `apt install cuda-toolkit-12-8` or rebuild with `-devel` image. |
| `-arch=sm_120a` rejected | CUDA < 12.8 | Re-rent on a newer machine; don't try to patch around it. |
| `RuntimeError: CUDA out of memory` during weight load | < 24 GB VRAM listing | RTX 5090 is 32 GB, but check `nvidia-smi` reports the full amount. |
| Kernel builds but `torch.ops.qwen_megakernel_C.decode` not found | Build name mismatch | Confirm `qwen_megakernel_tts/build.py` sets `name="qwen_megakernel_C"`. |
| Smoke test passes but RTF > 1 | Subtalker or codec is the bottleneck, not the kernel | Inspect per-stage timings in `bench_e2e.py` output. |
| KV hand-off explodes (shape mismatch) | HF cache layout differs from kernel expectation | Edit `MegakernelDecoder.set_kv_prefix` — print shapes, transpose, retry. |
