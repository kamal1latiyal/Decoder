# RTX 5090 Megakernel → Qwen3-TTS on Pipecat

Integrates [AlpinDale's qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
as the autoregressive-decode backend for Qwen3-TTS's talker, streaming real-time
speech into a Pipecat voice pipeline.

**Hardware**: RTX 5090 (sm_120 / Blackwell), CUDA 12.8+.

See [DESIGN.md](DESIGN.md) for the full architecture rationale, including the
parts of the talker that the kernel cannot do and why we keep HF in the loop
for text prefill + speaker conditioning.

---

## Architecture (one paragraph)

The Qwen3-TTS talker's text input goes through a separate 2048-dim embedding
projected to 1024, with MRoPE — none of which the megakernel implements. But
once you're past the text prefix and into the codec autoregressive loop, the
per-step backbone is shape-identical to Qwen3-0.6B. So we run the **text
prefill in HF** (one-time, ~50 ms) to warm the KV cache, then **swap in the
megakernel for the codec decode hot loop** (~1 ms/token at 12.5 tokens/sec of
audio). HF still runs the 5-layer subtalker (codebook predictor, ~2 ms/token)
and the codec decoder; their cost is fixed and small.

```
text  ─▶ HF prefill ─▶ [megakernel decode ↻] ─▶ subtalker ─▶ codec ─▶ PCM stream
                       ↑ the hot loop
```

---

## Quick start — one command on a fresh RTX 5090

```bash
# Optional: bring your own reference voice. If you don't, bootstrap downloads
# a 416 KB public-domain LJSpeech sample for you.
# export REF_AUDIO=/path/to/voice.wav
# export REF_TEXT="exact transcript of voice.wav"

bash scripts/bootstrap.sh
```

That single command performs steps [0]→[8]:

| Step | Action | ~time |
|---|---|---|
| 0 | Sanity check (`nvidia-smi`, `nvcc`, RTX 5090 detection) | 1 s |
| 1 | `install.sh` — clone kernel, install Python deps, JIT compile (`sm_120a`) | ~3 min |
| 2 | `download_models.py` — Qwen3-TTS weights from HF (~3 GB) | ~2 min |
| 3 | Fetch default reference voice (skipped if `REF_AUDIO` is set) | <5 s |
| 4 | `inspect_model.py --device cuda` — 11 static API checks, fail-fast | ~1 min |
| 5 | `smoke_test.py` — 9-stage verification (CUDA → kernel → KV parity → E2E) | ~5 min |
| 6 | Start FastAPI WebSocket server in background, wait for `/health` | ~30 s |
| 7 | Run TTFC / RTF / E2E benchmarks → `benchmarks/results/` | ~3 min |
| 8 | Print summary; server stays up for the Pipecat demo | — |

If any step fails the bootstrap exits non-zero with the failing stage label,
so you can re-run that step manually.

### Manual / step-by-step (if you want to bisect)

```bash
bash scripts/install.sh
python scripts/download_models.py
curl -fsSL https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav -o refs/voice.wav
export REF_AUDIO=$PWD/refs/voice.wav
export REF_TEXT="Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"

python scripts/inspect_model.py --device cuda    # cheap static check
python scripts/smoke_test.py                      # 9-stage fail-fast (incl. kernel↔HF parity)

python -m server.app --host 0.0.0.0 --port 8765 \
    --backend megakernel \
    --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT"
```

### Running the Pipecat voice loop (after the server is up)

```bash
export DEEPGRAM_API_KEY=...
export ANTHROPIC_API_KEY=...
python pipecat_integration/demo.py
```

### Quick WebSocket sanity check

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8765/synthesize") as ws:
        await ws.send(json.dumps({"text": "Hello, streaming TTS test."}))
        with open("out.raw", "wb") as f:
            async for msg in ws:
                if isinstance(msg, bytes):
                    f.write(msg)
                else:
                    print(json.loads(msg))

asyncio.run(main())
# ffplay -f s16le -ar 24000 -ac 1 out.raw
```

---

## Benchmarks

```bash
python benchmarks/bench_ttfc.py --runs 10
python benchmarks/bench_rtf.py  --runs 5
python benchmarks/bench_e2e.py
```

### Measured on a single RTX 5090 (Vast.ai, CUDA 13.0, May 2026)

| Metric                                    | Target   | Measured       |
|-------------------------------------------|----------|----------------|
| Kernel tok/s (isolated, 50-step warm)     | ~1000    | **1248**       |
| Kernel ↔ HF post-norm hidden cosine       | -        | **0.9998**     |
| Kernel ↔ HF argmax token agreement        | -        | **identical**  |
| TTFC (warm, median over 25 runs)          | < 90 ms  | **~260 ms**    |
| RTF (median over 9 runs)                  | < 0.3    | **0.746**      |
| Talker throughput (full pipeline)         | -        | ~17 tok/s      |
| Streaming chunk-by-chunk (vs buffered)    | required | **yes** (~320 ms cadence) |
| Audio quality (your voice cloned)         | clean    | **yes**        |

The kernel itself hits the throughput target with ~25 % headroom. The two
TTFC/RTF targets are missed by ~3× and ~2.5× respectively. **The bottleneck is
not the kernel** — it is HF's `GenerationMixin.generate()` invoked once per
audio frame for the subtalker. Each call has ~30–50 ms of Python orchestration
cost, multiplied by 12.5 frames per second of audio. A future revision that
bypasses `GenerationMixin` with a manual subtalker forward loop would close
~5–10× of the per-frame overhead, putting the pipeline back near the targets.

### What was changed to get these numbers

- Patched the upstream kernel's hardcoded `LDG_VOCAB_SIZE=151936` (Qwen3-0.6B
  text vocab) to `3072` (Qwen3-TTS codec head) via an **idempotent** in-place
  edit in `scripts/install.sh` plus a `-DLDG_VOCAB_SIZE` build flag. Without
  this the LM-head kernel reads ~148 K rows past the codec_head tensor → garbage.
- Replaced the original `subtalker.generate(..., output_hidden_states=True,
  return_dict_in_generate=True)` with the minimal-overhead version (we only
  use `.sequences`). The discarded 75 hidden-state tensors per frame were
  dominating wall time at 12.5 frames/sec.
- Fixed a sample-accounting bug in `tts/codec.py` that truncated megakernel
  audio after the first chunk: `_emitted_samples` was tracking the *current*
  decoded wav's length rather than cumulative emitted samples; with 4-frame
  overlap windows this caused every chunk after the first to emit zero.
- Capped `max_new_tokens` at 128 (≈10 s of audio) because greedy decoding in
  the kernel does not reliably emit the codec EOS token — uncapped runs
  would burn ~2 minutes of compute per request before hitting the upstream
  cap of 2048.
- Added a **CPU-only integration test** (`scripts/test_cpu_integration.py`)
  that loads the real `qwen_tts` model + your reference wav and exercises
  the entire HF-side path (monkey-patched prefill, `_extract_kv_for_kernel`,
  `CodePredictor.predict`) without CUDA. This caught the vocab hardcode and
  three API-mismatch bugs locally, before renting a single GPU minute.

### Benchmark methodology

- TTFC: `benchmarks/bench_ttfc.py --runs 5` — 5 representative sentences,
  5 runs each; first-run-of-server includes ~5 s of one-time CUDA init.
  Reported median is across the 24 warm runs.
- RTF: `benchmarks/bench_rtf.py --runs 3` — 3 runs × 3 sentences,
  wall-time / generated-audio-duration.
- E2E: `benchmarks/bench_e2e.py` — single composite run that also
  verifies streaming (multiple PCM chunks vs single bulk push).

Raw logs in `benchmarks/results/*.log` (uploaded with this repo).

---

## Backends

| Flag                          | Talker decode                  | Notes                          |
|-------------------------------|--------------------------------|--------------------------------|
| `--backend megakernel` (default) | `torch.ops.qwen_megakernel_C.decode` per step | Production target. |
| `--backend hf`                | `Qwen3TTSModel.generate(...)`  | Baseline. Verified correct. Not low-latency — HF generate returns the whole sequence at once. |

The `hf` backend exists for A/B comparison and for unblocking the rest of the
pipeline when the kernel needs debugging.

---

## Files & changes vs upstream

| File | Status |
|---|---|
| `csrc/kernel.cu` | **One-line patch** applied by `install.sh`: wraps the hardcoded `constexpr int LDG_VOCAB_SIZE = 151936` in `#ifndef … #endif` so a build flag controls vocab. Otherwise unmodified. |
| `csrc/torch_bindings.cpp` | Unmodified — symlinked from `AlpinDale/qwen_megakernel`. |
| `qwen_megakernel_tts/build.py` | JIT compile. Flags match upstream verbatim (LDG_PREFETCH_*, USE_UINT4, ATTENTION_VEC4, WEIGHT_LDCS, MLP_SMEM, `-arch=sm_120a`) **plus** `-DLDG_VOCAB_SIZE=3072` for the Qwen3-TTS codec head. |
| `qwen_megakernel_tts/model.py` | Extracts the talker's codec-side weights from a loaded `Qwen3TTSForConditionalGeneration` and packs them into the `LDGLayerWeights[28]` C struct layout. |
| `tts/talker.py` | `MegakernelDecoder` — single-step `torch.ops.qwen_megakernel_C.decode` wrapper. RoPE θ=1,000,000. KV-prefix injection for HF→kernel hand-off. |
| `tts/code_predictor.py` | Thin wrapper over the loaded model's subtalker (5 layers, HF). |
| `tts/codec.py` | Streaming wrapper over `model.speech_tokenizer` (12 Hz, 24 kHz). Carries 4-frame overlap, emits new samples only. |
| `tts/pipeline.py` | Orchestrator: HF prefill → megakernel decode loop → subtalker → codec → PCM stream. |
| `server/app.py` | FastAPI WS server. `--backend {megakernel,hf}` and `--ref-audio/--ref-text` for voice cloning. |
| `pipecat_integration/tts_service.py` | `MegakernelTTSService(TTSService)` — yields `TTSAudioRawFrame` per chunk. |
| `pipecat_integration/demo.py` | STT (Deepgram) → LLM (Claude Opus 4.7) → TTS → audio out. |

---

## Known limitations

| Limitation | Mitigation / status |
|---|---|
| **MRoPE not applied** in the kernel (uses unified positions) | Approximation only matters for multi-dim positions; single-channel speech is largely unaffected. To fix properly, patch the kernel's RoPE section. |
| **KV cache hand-off layout** — HF past_key_values shape must transpose cleanly to `[layer, kv_heads, seq, head_dim]` | Documented in `MegakernelDecoder.set_kv_prefix()`. Needs hardware run to confirm. |
| **Subtalker conditioning is one-step stale** | The kernel doesn't expose the talker's intermediate hidden state; predictor uses the pre-step hidden. Empirically tolerable for adjacent codec frames; verify on actual audio. |
| **Codec re-decode overhead** | We re-decode a 4-frame overlap per call to keep the codec's causal context warm; ~10 ms / call. Future fix: hook the codec's internal sliding-window state directly. |
| **Greedy decode only** | The kernel takes argmax. To add sampling we'd need to return logits instead of argmax (kernel change). |
| **`sm_120a` only** | Hard-coded in `build.py`. |

---

## Dependencies

- CUDA 12.8 + driver 575+ (Blackwell)
- PyTorch ≥ 2.7 with CUDA 12.8
- `transformers==4.57.3` (pinned by `qwen-tts`)
- `qwen-tts>=0.1.1`
- `pipecat-ai[websocket,deepgram,anthropic,silero]>=0.0.60`
- `fastapi`, `uvicorn[standard]`, `websockets`, `ninja`

See `requirements.txt`.
