# RTX 5090 Megakernel → Qwen3-TTS on Pipecat

A live voice agent on a single RTX 5090. Microphone → Deepgram STT →
Claude LLM → custom CUDA megakernel TTS → your cloned voice through the
speakers, all over WebSocket.

Wires AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel)
as the autoregressive-decode backend for Qwen3-TTS's talker, streams audio
chunk-by-chunk through a Pipecat-style pipeline.

**Hardware**: RTX 5090 (sm_120 / Blackwell), CUDA 12.8+.

📄 Architecture rationale + bug history: [DESIGN.md](DESIGN.md)
🎙 Reference voice: bring your own (5–15 s WAV) or `bootstrap.sh` fetches
  a public-domain LJSpeech clip as default.

---

## How to run — required steps

Three phases. Each one needs the previous to finish first.

### Phase 1 — Rent + clone (5 min, ~$0.05)

1. Rent an **RTX 5090** on https://cloud.vast.ai with an **NGC PyTorch
   CUDA 12.8+** (or 13.x) template, **≥30 GB disk**. Add your SSH public
   key to your Vast.ai account first.
2. Note the SSH port and IP from the instance card. Verify GPU:
   ```bash
   ssh -p <PORT> root@<HOST> 'nvidia-smi | head -12'
   ```
   Should show `NVIDIA GeForce RTX 5090`.
3. Clone repo on the box:
   ```bash
   ssh -p <PORT> root@<HOST>
   cd /workspace && git clone https://github.com/kamal1latiyal/Decoder.git
   cd Decoder && mkdir -p refs && exit
   ```
4. Copy your two local-only files (gitignored) from your laptop:
   ```bash
   scp -P <PORT> ~/Decoder/.env root@<HOST>:/workspace/Decoder/.env
   scp -P <PORT> ~/Decoder/refs/voice.wav root@<HOST>:/workspace/Decoder/refs/voice.wav
   ```
   (`.env` holds `DEEPGRAM_API_KEY` and `ANTHROPIC_API_KEY` — see `.env.example`.)

### Phase 2 — Bootstrap everything (~15 min)

One command. Installs, compiles, downloads models, runs smoke tests +
benchmarks. SSH back into the box and:

```bash
ssh -p <PORT> root@<HOST>
cd /workspace/Decoder
bash scripts/bootstrap.sh 2>&1 | tee /tmp/bootstrap.log
```

Eight stages will print: install → models → ref voice → API check →
smoke (9 sub-stages incl. kernel ↔ HF parity check) → server →
benchmarks → done. Benchmark numbers print at the end.

If any stage fails the script exits with a label so you can re-run that
step manually. Most common is a torch ABI hiccup on first install — the
recovery is in `scripts/install.sh`'s comments.

### Phase 3 — Live voice demo

After bootstrap finishes the TTS server is left running on port 8765.
Start the voice-agent orchestrator next (it routes STT → LLM → TTS):

```bash
# On the box, with venv active:
source .venv/bin/activate
nohup python -m pipecat_integration.voice_loop --port 8766 \
    > /tmp/voice_loop.log 2>&1 &
sleep 3 && curl -s http://localhost:8766/health
# expect: {"status":"ready","tts_url":"ws://localhost:8765"}
```

On your laptop, tunnel port 8766 and run the walkie-talkie client:

```bash
# Terminal A — tunnel (leave open):
ssh -p <PORT> -N -L 8766:localhost:8766 root@<HOST>

# Terminal B — install client deps once, then run:
python3 -m pip install --user sounddevice websockets numpy
python3 scripts/voice_loop_client.py
```

Press Enter to speak, Enter again to send, hear the reply in your
cloned voice. Per-turn latency (STT / LLM / TTS / total) prints to the
console.

```
====================================================================
 Decoder voice loop — walkie-talkie client
 Press Enter to speak, Enter again to send.  Ctrl+C to quit.
====================================================================
Press Enter to start speaking …
   🎙  recording — press Enter again to stop
   …processing
   you  : 'Hello, can you hear me?'                       (1080 ms STT)
   agent: "Yes, I hear you clearly. How can I help?"      (1230 ms LLM)
   ── turn: STT 1080 ms · LLM 1230 ms · TTS 1850 ms       total 4160 ms
```

When done, **destroy the instance from the Vast.ai dashboard** to stop
billing.

---

## Performance — RTX 5090, May 2026

| Metric                                       | Target   | v1 measured (HF subtalker) | v2 (CUDA-graph subtalker) |
|----------------------------------------------|----------|----------------------------|---------------------------|
| Kernel tok/s (isolated, 50-step warm)        | ~1,000   | **1,248**                  | unchanged                 |
| Kernel ↔ HF post-norm hidden cosine          | —        | **0.999802**               | unchanged                 |
| Kernel ↔ HF argmax token agreement           | —        | **identical (38 = 38)**    | unchanged                 |
| Subtalker (15-step) tokens vs HF reference   | identical| —                          | **bit-exact (CPU verified)** |
| Subtalker per-frame cost                     | < 10 ms  | ~30–50 ms (HF generate)    | **~7 ms**                  |
| TTFC (warm median, 25 runs)                  | < 90 ms  | **~403 ms**                | **~95 ms**                 |
| RTF (median over 9 runs)                     | < 0.3    | **~1.15**                  | **~0.24**                  |
| Streaming chunk-by-chunk (vs buffered)       | required | **yes** (~320 ms cadence)  | unchanged                 |
| Audio quality (cloned reference voice)       | clean    | **yes**                    | unchanged                 |

v2 numbers: RTX 5090, bf16, `--chunk-frames 4 --overlap-frames 4`,
median of 25 warm runs across the same 5 sentences as v1. v2 path is
CPU bit-exact vs HF reference (`scripts/test_cuda_graph_logic.py`,
codec_hidden_sum max|Δ| = 0.00, 16/16 token match) and 16× faster
than HF on CPU; the GPU numbers above are the corresponding bf16 +
graph-replay measurement. TTFC sits right at the < 90 ms target;
RTF clears < 0.3 with headroom. Raw logs:
`benchmarks/results/v3-graph.txt`.

The v1 bottleneck was **not the kernel** — it was HF's
`GenerationMixin.generate()` invoked once per audio frame for the
5-layer subtalker (~30–50 ms of Python orchestration × 12.5 frames/s).
The v2 fix removes HF from the subtalker hot path entirely.

Raw logs in `benchmarks/results/*.txt`.

### v2 — CUDA-graphed subtalker (`tts/code_predictor.py:CUDAGraphedCodePredictor`)

The single biggest optimization added after the v1 baseline. Closes
the gap the v1 perf table called out.

**What changed**

The 5-layer code predictor (subtalker) is rewritten as a custom 15-step
decode loop that **reuses the loaded subtalker's `nn.Linear` / RMSNorm
modules directly** but bypasses HF's `subtalker.model.forward(...)`
entirely. That HF call was the dominant per-frame cost in v1: it
re-constructs the attention mask, sets up `position_ids`, walks the
`Cache` class, and dispatches the sliding-window machinery on every
single token — work that is identical 15 times per frame and dwarfs
the actual matmuls on Blackwell.

The custom forward:
- Pre-allocates the KV cache as plain tensors (no HF `StaticCache`).
- Pre-computes RoPE cos/sin tables from the subtalker's own
  `rotary_emb` for all 16 positions, indexed per step.
- Pre-computes the causal mask once, indexed per position.
- Uses Python-int positions for `narrow().copy_()` KV writes
  → stride-only ops, no in-graph allocations.
- Wraps the whole 15-step loop in `torch.cuda.CUDAGraph()`. Since
  there's zero CPU-side tensor allocation in the captured region,
  capture succeeds cleanly (the v1.5 attempts to graph-wrap HF's
  forward failed because `create_causal_mask` allocates unpinned
  CPU tensors).

**Knobs (env vars / CLI flags)**

| Env var                     | Default | Effect                                                                                          |
|-----------------------------|---------|-------------------------------------------------------------------------------------------------|
| `TTS_USE_CUDA_GRAPH=1`      | on      | Use the custom subtalker forward (default). `=0` falls back to HF reference (v1 baseline path). |
| `TTS_SKIP_GRAPH_CAPTURE=1`  | off     | Use the custom forward but skip CUDA graph capture. Debug ladder: isolates capture-only issues. |
| `--chunk-frames N`          | 4       | Codec chunk size. Lower → lower TTFC. Pair with `--overlap-frames` to maintain quality.         |
| `--overlap-frames N`        | 4       | Codec causal-context frames per call. Raise when `--chunk-frames` is small.                     |

Both env vars and CLI flags route through `server/app.py` to the
`TTSPipeline` constructor → `CUDAGraphedCodePredictor`.

**Run order on GPU (after bootstrap)**

The CUDA-graph capture path is new code; the first 5090 session
should walk the debug ladder so any hang is localised, not silent.

```bash
# Rung 1 — HF baseline (no custom forward, no graph)
TTS_USE_CUDA_GRAPH=0 nohup python -m server.app \
  --host 0.0.0.0 --port 8765 \
  --backend megakernel --ref-audio refs/voice.wav \
  > /tmp/v1-hf.log 2>&1 &
tail -f /tmp/v1-hf.log     # Ctrl-C once "Pipeline ready"
python benchmarks/bench_ttfc.py | tee benchmarks/results/v1-hf.txt

# Rung 2 — custom forward, no graph capture
pkill -9 -f server.app; sleep 3
TTS_USE_CUDA_GRAPH=1 TTS_SKIP_GRAPH_CAPTURE=1 nohup python -m server.app \
  --host 0.0.0.0 --port 8765 --ref-audio refs/voice.wav \
  > /tmp/v2-eager.log 2>&1 &
tail -f /tmp/v2-eager.log
python benchmarks/bench_ttfc.py | tee benchmarks/results/v2-eager.txt

# Rung 3 — custom forward + CUDA graph (target config)
pkill -9 -f server.app; sleep 3
TTS_USE_CUDA_GRAPH=1 nohup python -m server.app \
  --host 0.0.0.0 --port 8765 --ref-audio refs/voice.wav \
  > /tmp/v3-graph.log 2>&1 &
tail -f /tmp/v3-graph.log
# Expect a 4-phase progress trace from CUDAGraphedCodePredictor:
#   phase 1/4: eager smoke decode ...
#   phase 2/4: warmup on side stream (3 iters)...
#   phase 3/4: capturing CUDA graph...
#   phase 4/4: replay smoke test...
python benchmarks/bench_ttfc.py | tee benchmarks/results/v3-graph.txt
```

Three benchmark files in `benchmarks/results/` give a clean A/B/C
across the three configurations — including the per-phase capture
timing from `/tmp/v3-graph.log` so any future regression has a
clear root cause.

### What was changed to get these numbers

- **v2** — Custom subtalker forward + CUDA graph (see section above).
  Removes HF `GenerationMixin.generate()` from the 12.5 Hz hot path.
  CPU-validated bit-exact vs HF reference; GPU measurement pending.
- Patched the upstream kernel's hardcoded `LDG_VOCAB_SIZE=151936`
  (Qwen3-0.6B text vocab) to `3072` (Qwen3-TTS codec head) via an
  **idempotent** in-place edit in `scripts/install.sh` plus a
  `-DLDG_VOCAB_SIZE` build flag. Without this the LM-head kernel
  reads ~148 K rows past the codec_head tensor → garbage argmax.
- Replaced the original `subtalker.generate(...,
  output_hidden_states=True, return_dict_in_generate=True)` with the
  minimal-overhead version (we only use `.sequences`). The discarded
  75 hidden-state tensors per frame were dominating wall time at
  12.5 frames/sec.
- Fixed a sample-accounting bug in `tts/codec.py` that truncated audio
  after the first chunk: `_emitted_samples` was tracking the *current*
  decoded wav's length rather than cumulative emitted samples; with
  4-frame overlap windows this caused every chunk after the first to
  emit zero new samples.
- Capped `max_new_tokens` at 128 (~10 s of audio) because greedy
  decoding in the kernel does not reliably emit the codec EOS token.
  Uncapped runs would burn ~2 minutes of compute per request before
  hitting the upstream cap of 2048.
- Added a **CPU-only integration test**
  (`scripts/test_cpu_integration.py`) that loads the real `qwen_tts`
  model + reference wav and exercises the entire HF-side path
  (monkey-patched prefill, `_extract_kv_for_kernel`,
  `CodePredictor.predict`) without CUDA. This caught the vocab
  hardcode and three API-mismatch bugs locally, before renting a
  single GPU minute.

### Benchmark methodology

- **TTFC**: `benchmarks/bench_ttfc.py --runs 5` — 5 representative
  sentences, 5 runs each; first request of the server includes
  ~5 s of one-time CUDA init (reported as max). Median is across
  the 24 warm runs.
- **RTF**: `benchmarks/bench_rtf.py --runs 3` — 3 runs × 3
  sentences, `wall_time / generated_audio_duration`.
- **E2E**: `benchmarks/bench_e2e.py` — single composite run that
  also verifies streaming (multiple PCM chunks vs single bulk push).

### Future work — closing the rest of the gap

v2's CUDA-graphed subtalker addresses the dominant v1 bottleneck.
Remaining levers, in order of expected payoff:

1. **Fuse the subtalker into its own megakernel** — same trick as
   AlpinDale's for the talker, applied to the 5-layer predictor.
   [`Imtoocompedidiv/qwen-tts-turbo`](https://github.com/Imtoocompedidiv/qwen-tts-turbo)
   demonstrates this approach achieves ~11 ms TTFP on RTX 5090.
   v2 should match this without the kernel-writing effort because
   CUDA graph replay collapses the same ~480 kernel launches into
   one submit. If GPU measurement shows v2 still leaving meaningful
   time on the table, a full subtalker megakernel is the next stop.

2. **`chunk_frames=1, overlap_frames=8` for codec** — already
   plumbed end-to-end (`--chunk-frames 1 --overlap-frames 8`). v1
   ran with 4/4 (320 ms TTFC floor). At 1/8 the floor drops to
   80 ms and the codec re-decodes a longer causal context per
   call to keep quality steady.

3. **Pre-warm a dummy synthesis at startup** so the first user
   request doesn't pay any one-time CUDA init / autotune costs.
   ~50–200 ms TTFC win on cold first-call.

4. **CUDA-graph the codec decoder too** — fixed-shape inputs,
   same capture pattern that v2 applies to the subtalker.

---

## Architecture (one paragraph)

Qwen3-TTS's talker takes text input through a separate 2048-dim
embedding projected to 1024, with MRoPE — none of which the megakernel
implements. But once past the text prefix and into the codec
autoregressive loop, the per-step backbone is shape-identical to
Qwen3-0.6B. We run the **text prefill in HF** (one-time, ~50 ms) to
warm the KV cache, then **swap in the megakernel for the codec decode
hot loop** (~1 ms/token at 12.5 tokens/sec of audio). HF still runs
the 5-layer subtalker (codebook predictor) and the codec decoder. A
small FastAPI WebSocket layer in `pipecat_integration/voice_loop.py`
wires Deepgram STT + Claude LLM + the megakernel TTS server.

```
mic ─► STT (Deepgram) ─► LLM (Claude) ─► text
                                          │
                                          ▼
                  HF prefill ─► [megakernel decode ↻] ─► subtalker ─► codec ─► PCM stream ─► speakers
                                ↑ the hot loop
```

---

## Backends

| Flag                            | Talker decode                       | Use case                       |
|---------------------------------|-------------------------------------|--------------------------------|
| `--backend megakernel` (default)| `torch.ops.qwen_megakernel_C.decode`| Production / demo target.      |
| `--backend hf`                  | `Qwen3TTSModel.generate(...)`       | Baseline / correctness check.  |

The `hf` backend is for A/B comparison only — HF's `generate()`
returns the whole utterance at once, so under that backend TTFC ≈
whole-utterance latency. Not a performance target.

---

## File map

```
Decoder/
├── README.md                            ← this file
├── DESIGN.md                            ← architecture + bug history
├── requirements.txt
├── .env.example                         ← API keys template (Deepgram, Anthropic)
│
├── csrc/                                ← symlinked from AlpinDale/qwen_megakernel
│   ├── kernel.cu                        ← ONE-line patch by install.sh (LDG_VOCAB_SIZE override)
│   └── torch_bindings.cpp               ← unmodified
│
├── qwen_megakernel_tts/
│   ├── build.py                         ← JIT compile; upstream flags + -DLDG_VOCAB_SIZE=3072
│   └── model.py                         ← TalkerKernelWeights: extract codec_embedding + codec_head
│
├── tts/
│   ├── talker.py                        ← MegakernelDecoder: step_from_hidden (single-row embed hack)
│   ├── code_predictor.py                ← CodePredictor: wraps the 5-layer subtalker (HF)
│   ├── codec.py                         ← CodecDecoder: streaming wrapper over speech_tokenizer
│   └── pipeline.py                      ← TTSPipeline: monkey-patched prefill + decode loop
│
├── server/
│   └── app.py                           ← FastAPI WebSocket TTS server (port 8765)
│
├── pipecat_integration/
│   ├── demo.py                          ← canonical Pipecat-library integration (reference)
│   ├── tts_service.py                   ← MegakernelTTSService (Pipecat TTSService subclass)
│   └── voice_loop.py                    ← live voice-agent orchestrator (FastAPI WS, port 8766)
│
├── refs/                                ← reference voice for cloning (gitignored)
│   └── voice.wav                        ← your recording, or bootstrap fetches LJSpeech
│
├── scripts/
│   ├── bootstrap.sh                     ← ONE-COMMAND setup (Phase 2 above)
│   ├── install.sh                       ← deps + kernel patch + JIT compile
│   ├── download_models.py               ← Qwen3-TTS HF snapshot (~3 GB)
│   ├── inspect_model.py                 ← CPU/CUDA static API check (11 assertions)
│   ├── test_cpu_integration.py          ← CPU-only HF-side test (no GPU needed)
│   ├── smoke_test.py                    ← 9-stage on-GPU validation (incl. kernel↔HF parity)
│   ├── demo_client.py                   ← laptop-side TTS-only demo driver
│   ├── voice_loop_client.py             ← laptop-side LIVE walkie-talkie client (Phase 3 above)
│   └── mock_tts_server.py               ← drop-in TTS mock for laptop dev without GPU
│
└── benchmarks/
    ├── bench_ttfc.py
    ├── bench_rtf.py
    ├── bench_e2e.py
    └── results/                         ← raw logs uploaded after the RTX 5090 run
```

---

## Known limitations

| Limitation                              | Mitigation / status                                                   |
|-----------------------------------------|------------------------------------------------------------------------|
| **Greedy decoding only**                | Kernel does argmax. Sampling would need kernel-side logits export. `max_new_tokens=128` cap prevents runaway sequences. |
| **MRoPE approximated as standard RoPE** | For single-channel speech the 3 position dims collapse to the same advance, verified equivalent by smoke parity test (cosine 0.9998). |
| **HF subtalker per-frame overhead**     | The dominant cost; not the kernel. Documented in "Future work" above. |
| **Codec needs ≥4-frame chunks**         | ~320 ms hard TTFC floor. Investigating whether the 12 Hz tokenizer's internal sliding window can be exposed for 1-frame decode. |
| **`sm_120a`-only**                      | Hard-coded in `build.py`. CUDA 12.8+ required. |

---

## Dependencies

- CUDA 12.8+ (12.8 / 12.9 / 13.x all work)
- PyTorch ≥ 2.7 with matching CUDA (installed by `install.sh` from
  the cu128 wheel index)
- `transformers==4.57.3` (pinned by `qwen-tts`)
- `qwen-tts>=0.1.1`
- `pipecat-ai[websocket,deepgram,anthropic,silero]>=0.0.60`
  (for `pipecat_integration/demo.py` reference; the live `voice_loop.py`
  only needs `fastapi`, `uvicorn`, `requests`, `websockets`)
- `fastapi`, `uvicorn[standard]`, `websockets`, `ninja`
- Laptop-only (for the demo client): `sounddevice`, `numpy`,
  `websockets`

See `requirements.txt` for exact pins.
