# System Design — RTX 5090 Megakernel → Qwen3-TTS on Pipecat

## 1. Problem

Use AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel)
(~1,000 tok/s for Qwen3-0.6B on RTX 5090) as the autoregressive-decode backend
for Qwen3-TTS's talker, exposed as a streaming TTS service inside a Pipecat
voice pipeline.

---

## 2. Architecture match — what's actually compatible

The first draft of this design (commit history) claimed the megakernel was a
near drop-in for the talker. Local CPU-side inspection of the real
`qwen_tts==0.1.x` package source (verified 2026-05-21 against
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` and `scripts/inspect_model.py`) shows it
isn't. Honest comparison:

| Parameter                       | Megakernel (Qwen3-0.6B) | Qwen3-TTS Talker            | Match |
|---------------------------------|------------------------|-----------------------------|-------|
| `hidden_size`                   | 1024                   | 1024                        | ✓ |
| `num_hidden_layers`             | 28                     | 28                          | ✓ |
| `num_attention_heads`           | 16                     | 16                          | ✓ |
| `num_key_value_heads`           | 8 (GQA)                | 8 (GQA)                     | ✓ |
| `head_dim`                      | 128                    | 128                         | ✓ |
| `intermediate_size`             | 3072                   | 3072                        | ✓ |
| `rope_theta`                    | 10,000                 | **1,000,000**               | ✗ Python-side fix |
| `rope_scaling`                  | standard               | **MRoPE [24,20,20]**        | ✗ approximation |
| Input embedding                 | text vocab 151,936     | **text + codec; text hidden = 2048**  | ✗ HF handles |
| Output head                     | vocab 151,936          | **codec vocab 3,072**       | ✓ kernel infers from tensor shape |

Three consequences (all confirmed by reading `qwen_tts` source 2026-05-21):

1. **The megakernel cannot do text prefill.** The talker projects text
   embeddings from `text_hidden_size=2048 → hidden_size=1024` (`text_projection`)
   before entering the transformer stack — the kernel has no such projection.
   We do prefill in HF (one forward through `talker.model`) and hand the warm
   KV cache to the kernel.

2. **The codec decode loop is NOT just embed(token_id) → transformer.** From
   `Qwen3TTSTalkerForConditionalGeneration.forward` (modeling_qwen3_tts.py
   lines 1665–1692), the talker's per-step input is

      inputs_embeds = Σᵢ embedᵢ(groupᵢ) + trailing_text_hidden[step]

   — the SUM of 16 codec-token embeddings (talker emits group-0; subtalker
   emits groups 1..15) plus a projected text-hidden vector for the current
   step. This 1024-d vector cannot be represented as `embed[token_id]` for
   any token_id, so the kernel's `decode(token_id, ...)` signature doesn't
   fit directly.

   **Fix without kernel changes**: the kernel does
   `embed_row = embed_weight + token_id * HIDDEN_SIZE`. We pass a 1-row bf16
   embed table containing our pre-computed input hidden and `token_id=0`,
   so the lookup lands on our injected vector. See
   `MegakernelDecoder.step_from_hidden()` in `tts/talker.py`. The CUDA code
   path is identical to a normal embed lookup — same memory traffic, same
   address calculation, same downstream layers; we're just abusing the
   embed table as a 1-entry hidden cache.

3. **The codec decode HIDDEN STATE is exposed by the kernel.** The kernel
   writes the post-norm hidden into `g_normalized` (float32 [HIDDEN_SIZE])
   before the LM head matmul — our `self.norm_out` tensor. We return a
   bf16 view of it from `step_from_hidden()`, and the subtalker uses it as
   `past_hidden` for the NEXT frame. This fixes the "subtalker conditioning
   is one-step stale" limitation that the initial design listed.

The 1-row table is bf16 like the real codec embed; the LM head is the talker's
`codec_head` (NOT `lm_head` — that's the text-vocab head we ignore).

**Vocab size IS compile-time-baked into the kernel.** Upstream hardcodes
`constexpr int LDG_VOCAB_SIZE = 151936;` (Qwen3-0.6B text vocab). We patch
upstream via `install.sh` (an idempotent perl edit that wraps the constexpr in
`#ifndef`), then pass `-DLDG_VOCAB_SIZE=3072` from `qwen_megakernel_tts/build.py`.
This is the only deviation from the upstream kernel source. Without it the LM-head
kernel would OOB-read 148,864 rows past the codec_head tensor and return
garbage argmax — caught locally before renting the 5090, see commit history.

---

## 3. Pipeline

```
┌────────────────────────── Pipecat Voice Pipeline ────────────────────────┐
│  Mic → Deepgram STT → LLM → MegakernelTTSService → Speaker               │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │ ws JSON {text}
              ┌───────────────────▼──────────────────────────────┐
              │   FastAPI WebSocket server  (server/app.py)      │
              │                                                  │
              │  text                                            │
              │   │                                              │
              │   ▼  HF talker.forward(prefill_text)             │
              │   ├── builds KV cache  ─────┐                    │
              │   ├── first codec token     │                    │
              │   ▼                         │                    │
              │  MegakernelDecoder ◄────────┘   ← KV hand-off    │
              │   │  torch.ops.qwen_megakernel_C.decode          │
              │   │  ~1 ms / token                               │
              │   ▼                                              │
              │  Subtalker.generate_one_frame (HF, 5 layers)     │
              │   │  group-0 → groups 1..15  (~2 ms)             │
              │   ▼                                              │
              │  Frame buffer (≥4 frames = 320 ms)               │
              │   ▼                                              │
              │  CodecDecoder (Qwen3-TTS-Tokenizer-12Hz)         │
              │   │  PCM @ 24 kHz mono int16                     │
              │   ▼ yield ws.binary  →  TTSAudioRawFrame         │
              └──────────────────────────────────────────────────┘
```

### Stage 1 — text prefill (HF, one-time per utterance)
- The wrapper's text/codec prefix construction is fragile (handles
  `voice_clone_prompt`, language id, codec_think/codec_pad/codec_bos
  special tokens, speaker x-vector injection — ~80 lines in
  `Qwen3TTSForConditionalGeneration.generate`). Re-implementing that
  ourselves would silently drift on any qwen_tts bump.

- Instead: **monkey-patch `model.talker.generate`** to do a single
  `talker.model.forward(inputs_embeds=…, use_cache=True)`, then raise a
  sentinel `_PrefillDone` exception. We call
  `wrapper.generate_voice_clone(text=text, voice_clone_prompt=…, do_sample=False, max_new_tokens=1)`
  and catch the sentinel. The wrapper handles all the input-embeds
  construction; we just intercept right before generation would loop.
  See `tts/pipeline.py: _hf_prefill`.

- Captures: `(first_token, kv_cache, last_hidden, trailing_text_hidden, tts_pad_embed)`.
- `last_hidden_state` from `talker.model.forward` is already post-norm
  (`Qwen3TTSTalkerModel.forward` applies `self.norm` before returning,
  modeling_qwen3_tts.py:1550). That matches what the megakernel's
  `g_normalized` produces, so the two paths are layout-compatible.
- Cost: ~30–80 ms for a typical 30–60 text-token prompt.

### Stage 2 — megakernel codec decode (hot loop, 12.5 Hz of audio)
- `MegakernelDecoder.step_from_hidden(input_hidden)` calls
  `torch.ops.qwen_megakernel_C.decode(...)` with `embed_weight = 1×1024 bf16
  row containing the caller's hidden`, `token_id = 0`, plus the codec-vocab
  `codec_head` weight and the warmed KV cache. Returns
  `(next_codec_token_id, post_norm_hidden_bf16[1024])`.
- Target: ≈1 ms / step (same as upstream Qwen3-0.6B since the per-step
  compute is identical — the embed lookup is the same 2 KB read).
- 1 s of audio = 12.5 talker steps ⇒ ~12.5 ms of kernel work.

### Stage 3 — code predictor (HF, 5 layers, ~2 ms/token)
- `model.talker.code_predictor.generate(inputs_embeds=cat(past_hidden, embed(group0)), max_new_tokens=15, output_hidden_states=True, return_dict_in_generate=True)`
  emits groups 1..15 of the current frame.
- `past_hidden` is the talker post-norm hidden from the current kernel step
  — `MegakernelDecoder.step_from_hidden`'s second return value.
- The orchestrator builds the next kernel input from the subtalker's frame:

      codec_hidden_sum = embed(group0) + Σᵢ₌₁..₁₅ embedᵢ(groupᵢ)   # CodePredictor returns this
      next_input       = codec_hidden_sum + (trailing_text_hidden[step] OR tts_pad_embed)

  exactly mirroring modeling_qwen3_tts.py:1683–1692. Not a throughput
  bottleneck.

### Stage 4 — codec decode (streaming with overlap)
- `Qwen3-TTS-Tokenizer-12Hz.decode([{"audio_codes": frames}])` produces wav.
- Minimum decode window is 4 frames = 320 ms — the hard TTFC floor for audio
  out from the codec.
- We carry the last `OVERLAP_FRAMES=4` frames between calls and re-decode the
  overlap, then emit only the new samples. Sample-accurate truncation against
  cumulative emitted-sample count keeps the stream gap-free.

---

## 4. Kernel — what we built, what we changed

The CUDA source is **near-unmodified upstream**. One single-line patch is
applied by `scripts/install.sh` (idempotent perl edit on the freshly cloned
upstream kernel): wrap the `constexpr int LDG_VOCAB_SIZE = 151936` in an
`#ifndef … #endif` so the build flag controls vocab. `csrc/torch_bindings.cpp`
is untouched.

The build flags in `qwen_megakernel_tts/build.py` mirror upstream verbatim
(LDG_PREFETCH_*, USE_UINT4, ATTENTION_VEC4, WEIGHT_LDCS, MLP_SMEM,
`-arch=sm_120a`) **plus** `-DLDG_VOCAB_SIZE=3072` for the Qwen3-TTS codec
head. Without that override the LM-head kernel would read 148K rows past
the codec_head tensor → garbage argmax.

The ops are exposed via TORCH_LIBRARY — we call them as
`torch.ops.qwen_megakernel_C.decode(...)`, not via the loaded module's
attribute.

### 4.1 RoPE
RoPE cos/sin tables are recomputed in Python with `theta=1_000_000` (Qwen3-TTS
talker) instead of `10_000` (Qwen3-0.6B). The kernel just reads the table.

### 4.2 MRoPE (limitation, acknowledged)
The talker config specifies MRoPE with `mrope_section=[24,20,20]`. The
megakernel applies a single position frequency per dim. For single-channel
speech (no image / video position dims) the three sections collapse to the
same per-step advance, so this matters less than it looks — but it's still an
approximation, not a faithful implementation. We document this and move on.

### 4.3 Buffer sizes
`bmax_vals/idxs` are sized 4096 (safe over-allocation; LM-head grid is
LDG_LM_NUM_BLOCKS=1280 blocks, each writes one entry).

### 4.4 LDG_VOCAB_SIZE override
See section 2 — the kernel's `LDG_VOCAB_SIZE` constexpr is patched in
`install.sh` to be overridable, and `qwen_megakernel_tts/build.py` passes
`-DLDG_VOCAB_SIZE=3072` for the codec vocab.

### 4.5 Speaker conditioning
The official `Qwen3TTSModel.create_voice_clone_prompt(ref_audio, ref_text)`
extracts the speaker x-vector (1024-dim) via the model's
`extract_speaker_embedding`. We pass the resulting prompt list into
`wrapper.generate_voice_clone(..., voice_clone_prompt=...)` during prefill so
the KV cache is correctly conditioned with the speaker identity. If `ref_text`
is omitted, the wrapper auto-selects `x_vector_only_mode=True` (speaker
embedding only, no in-context-learning).

### 4.6 Monkey-patched prefill (HF → kernel hand-off)
The HF wrapper's `generate_voice_clone()` builds ~80 lines of input embeddings
(text projection + codec prefix tokens + speaker embedding + special tokens),
then calls `model.talker.generate(inputs_embeds=…)`. Re-implementing this
ourselves would drift on every `qwen_tts` version bump. So `_hf_prefill` in
`tts/pipeline.py` **monkey-patches `model.talker.generate`** for one invocation:
the replacement runs `talker.model.forward(inputs_embeds=…, use_cache=True)`
once, applies `codec_head` to the last hidden, then raises a sentinel exception
carrying `(first_token, kv_cache, last_hidden, trailing_text_hidden, tts_pad_embed)`.
The wrapper does all the heavy lifting; we just intercept the moment generation
would loop.

### 4.7 Greedy decoding (limitation)
The kernel's LM head produces argmax tokens — no sampling. HF normally drives
the talker with `do_sample=True, temperature=0.9` to provide diversity that
helps the model find natural EOS points. Greedy decoding sometimes fails to
emit the codec EOS token, so we cap `max_new_tokens=128` (~10 s of audio) as
a hard ceiling. Adding sampling would require kernel-side logits-export +
sampling — out of scope for this take-home.

---

## 5. Streaming

```python
async for pcm in pipeline.synthesize(text):
    await ws.send_bytes(pcm)
```

`pipeline.synthesize()` is an async generator. Each yield is one chunk
(default 320 ms / 7,680 bytes, observed cadence ~320 ms in benchmarks) and is
sent over the WebSocket immediately. We `await asyncio.sleep(0)` between decode
steps so the event loop can dispatch audio while the GPU runs the next kernel
call — GPU compute and audio dispatch interleave.

`tts/codec.py` decodes `history + new_frames` together so the codec has its
causal context, then emits **only the trailing window** (the last
`len(new_frames) * SAMPLES_PER_FRAME` samples). An earlier version tracked
absolute sample counts cumulatively — that was wrong, because the codec's
output for `history + new` is a fresh decode each call, not a continuation
of prior wavs. Verified by inspection of `bench_e2e.py` output:
32 chunks × 7,680 bytes = 491,520 bytes = 10.24 s of audio, no gaps.

---

## 6. Backends

| Backend       | Talker decode                  | Use case                       |
|---------------|--------------------------------|--------------------------------|
| `megakernel`  | `torch.ops.qwen_megakernel_C`  | Production (RTX 5090).         |
| `hf`          | `Qwen3TTSModel.generate(...)`  | Baseline / correctness check.  |

The `hf` backend exists so that if the kernel build fails, KV layout doesn't
match, or you want an A/B sanity check, the pipeline still works end-to-end.
It is NOT a performance target — HF generate() returns the full sequence in
one call, so under that backend TTFC ≈ whole-utterance latency.

---

## 7. WebSocket protocol

```
GET  /health      → { status, cuda_device, cuda_memory_gb, startup_time_s }
GET  /metrics     → last synthesis metrics (TTFC, RTF, tok/s)
WS   /synthesize  → JSON in, binary PCM out, JSON done out

  Client → JSON  {"text": "...", "speaker": "default"}
  Server → bytes int16 LE PCM @ 24 kHz mono   (multiple messages)
  Server → JSON  {"type": "done", "metrics": {...}}
```

GPU is single-tenanted; concurrent requests queue on a per-server asyncio
lock. Cold-load (~30 s) once at startup.

---

## 8. Performance — measured, not estimated

Numbers from the RTX 5090 run (Vast.ai, CUDA 13.0, May 2026). Raw logs in
`benchmarks/results/*.log`.

| Metric                                | Target     | Measured              |
|---------------------------------------|------------|-----------------------|
| Kernel tok/s (isolated, 50 warm steps)| ~1,000     | **1,248** (25 % over) |
| Kernel ↔ HF post-norm hidden cosine   | —          | **0.999802**          |
| Kernel ↔ HF argmax token agreement    | —          | **identical**         |
| TTFC (warm median, 25 runs)           | < 90 ms    | **~260 ms**           |
| RTF (median, 9 runs)                  | < 0.3      | **0.746**             |
| RTF std-dev                           | —          | 0.003 (very stable)   |
| Talker tok/s (full pipeline)          | —          | ~17                   |
| Streaming chunk cadence               | required   | ~320 ms (8 chunks/s)  |
| Cloned-voice audio quality            | acceptable | clean, your voice     |

The kernel itself hits the throughput target with headroom. The end-to-end
metrics miss the TTFC and RTF targets by ~3× and ~2.5× respectively. **The
bottleneck is not the kernel** — it is HF's `GenerationMixin.generate()`
called once per audio frame for the 5-layer subtalker (~30–50 ms of Python
orchestration cost × 12.5 frames/s ≈ 0.4–0.6 s of overhead per second of
audio). The kernel call itself adds only ~1–2 ms per step.

Where the budget for one second of audio actually goes:

| Stage                                    | Measured wall time per 1 s audio |
|------------------------------------------|----------------------------------|
| HF talker prefill (one-time, amortised)  | ~50 ms (one-time per request)    |
| Megakernel talker decode (12.5 steps)    | ~15–25 ms                        |
| HF subtalker per frame (12.5 × ~50 ms)   | ~625 ms                          |
| Codec decode (every 4 frames)            | ~50 ms                           |
| Asyncio + tensor ops orchestration       | ~50 ms                           |
| **Observed total per 1 s audio**         | **~750 ms (RTF ≈ 0.75)**         |

A subtalker megakernel would replace the ~625 ms subtalker line with
something like ~25 ms (mirroring the talker kernel ratio), bringing total
per 1 s audio to ~150 ms (RTF ~0.15) — putting the pipeline well under
the target. See README "Future work" for details.

---

## 9. Validation status + known limitations

### Verified on hardware (RTX 5090, May 2026)

| Item | Status |
|---|---|
| KV cache hand-off layout (HF → kernel) | **Verified** by smoke_test stage 6: identical argmax tokens (HF=38, kernel=38), post-norm hidden cosine 0.999802. Math is equivalent. |
| Subtalker hidden-state staleness | **Fixed**. Kernel exposes post-norm hidden via `g_normalized` buffer (=`self.norm_out`); `step_from_hidden` returns it bf16. Subtalker sees the current step's hidden. |
| Streaming chunk-by-chunk (not buffered) | **Verified** by bench_e2e.py: 32 chunks × 7,680 bytes per 10.24 s utterance, ~320 ms cadence. |
| Audio quality | **Verified** by listening to `smoke_test_mega.wav` (cloned reference voice) and the 3 utterances from `scripts/demo_client.py`. |
| Benchmarks | **Produced**. See section 8 + `benchmarks/results/`. |
| Demo recording | **Produced**. TTS-driven demo using `scripts/demo_client.py` over SSH tunnel to the box. |

### Known limitations (documented, not blocking)

| Item | Why it exists | Future fix |
|---|---|---|
| MRoPE approximated as standard RoPE | Upstream kernel uses split-half RoPE; talker config uses `mrope_section=[24,20,20]`. For single-channel speech all three position dims advance identically so MRoPE collapses to standard RoPE — verified by cos 0.9998 in the parity test. | Patch kernel's RoPE section indexing if multi-channel ever matters. |
| Greedy decoding (no sampling) | Kernel's LM head produces argmax only. With greedy, the model sometimes fails to emit EOS naturally. We cap `max_new_tokens=128` (~10 s audio) as a hard ceiling. | Add kernel-side sampling (return logits + sample on host, or fuse sampling into kernel). |
| Codec needs ≥4 frames per decode | The 12 Hz tokenizer's internal sliding window can't decode fewer than 4 frames. Forces TTFC floor ≈ 320 ms even with a free synthesis path. | Investigate exposing `qwen_tts.core.tokenizer_12hz` internal state for 1-frame incremental decode. |
| Default-voice (no ref audio) | Both backends require a reference voice file. `qwen_tts` doesn't ship a default x-vector seed. | Ship a stock reference clip (LJSpeech sample auto-downloaded by `bootstrap.sh`). |
| HF GenerationMixin per-frame overhead | The 5-layer subtalker is called via `HF.generate(...)` once per audio frame — ~30–50 ms of Python boilerplate per call dominates RTF. | Bypass with manual subtalker forward loop (~3–5× speedup, no kernel work) or fuse a subtalker megakernel (~10× speedup, à la qwen-tts-turbo). |

### Bugs found + fixed post-hardware-run

These were caught only after running the full pipeline on the RTX 5090 — kept here as a record:

1. **`subtalker.generate(output_hidden_states=True)`** allocated 75 unused tensors per frame (5 layers × 15 generation steps). At 12.5 frames/s this dominated wall time. Removed the flag.
2. **`codec.decode` sample-tracking bug**: `_emitted_samples` tracked the cumulative *wav* length rather than emitted samples. After 2 codec calls with 4-frame overlap, every subsequent decode emitted 0 new samples → audio truncated silently at 8 frames. Fixed by slicing the trailing `len(frames) * SAMPLES_PER_FRAME` from each decode instead.
3. **`max_new_tokens=2048` cap was too high** given that greedy decoding often fails to emit EOS. Loops would run for ~2 minutes per request before hitting cap. Lowered to 128.
4. **`install.sh` `--system-site-packages` + ABI drift**: an attempted "preserve NGC PyTorch" optimisation caused pip to install a torchaudio that didn't match NGC's torch ABI (undefined symbol `torch_dtype_float4_e2m1fn_x2`). Reverted to clean venv + cu128 wheel reinstall.

---

## 10. File map

```
Decoder/
├── README.md                            ← user-facing: quick start + benchmarks + future-work
├── DESIGN.md                            ← this file: architecture rationale + bug history
├── requirements.txt
├── .env.example                         ← API keys template (DEEPGRAM_API_KEY, ANTHROPIC_API_KEY)
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
│   └── app.py                           ← FastAPI WebSocket TTS server
│
├── pipecat_integration/
│   ├── tts_service.py                   ← MegakernelTTSService (pipecat TTSService subclass)
│   └── demo.py                          ← STT (Deepgram) → LLM (Claude) → TTS → audio out
│
├── refs/                                ← reference voice for cloning (gitignored)
│   └── voice.wav                        ← provide your own, or bootstrap.sh fetches LJSpeech
│
├── scripts/
│   ├── bootstrap.sh                     ← ONE-COMMAND setup: install → benchmarks (~15 min)
│   ├── install.sh                       ← deps + kernel patch + JIT compile
│   ├── download_models.py               ← Qwen3-TTS HF model snapshot (~3 GB)
│   ├── inspect_model.py                 ← CPU/CUDA static API check (11 assertions)
│   ├── test_cpu_integration.py          ← CPU-only HF-side end-to-end test (no GPU needed)
│   ├── smoke_test.py                    ← 9-stage on-GPU validation (incl. kernel↔HF parity)
│   └── demo_client.py                   ← laptop-side demo driver (mic-out + playback + metrics)
│
└── benchmarks/
    ├── bench_ttfc.py
    ├── bench_rtf.py
    ├── bench_e2e.py
    └── results/                         ← raw logs from RTX 5090 run
        ├── ttfc.log
        ├── rtf.log
        └── e2e.log
```
