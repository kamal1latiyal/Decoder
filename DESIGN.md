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

The CUDA source (`csrc/kernel.cu`, `csrc/torch_bindings.cpp`) is **unmodified**
upstream. The build flags in `qwen_megakernel_tts/build.py` mirror upstream
verbatim (LDG_PREFETCH_*, USE_UINT4, ATTENTION_VEC4, WEIGHT_LDCS, MLP_SMEM,
`-arch=sm_120a`).

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
`extract_speaker_embedding`. We pass the resulting prompt dict into
`generate(..., voice_clone_prompt=...)` during prefill so the KV cache is
correctly conditioned. If no reference audio is given, the prompt is built
with `x_vector_only_mode=True` using the model's default identity.

---

## 5. Streaming

```python
async for pcm in pipeline.synthesize(text):
    await ws.send_bytes(pcm)
```

`pipeline.synthesize()` is an async generator. Each yield is one chunk
(default 320 ms / 7,680 bytes) and is sent to Pipecat immediately. We
`await asyncio.sleep(0)` between decode steps so the event loop can dispatch
audio while the GPU runs the next kernel call — GPU compute and audio
dispatch interleave.

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

## 8. Performance budget

| Stage                        | Per token   | For 1s audio (12.5 frames) |
|------------------------------|-------------|----------------------------|
| Text prefill (~40 tokens, HF)| —           | ~50 ms (one-time)          |
| Megakernel talker decode     | ~1.0 ms     | ~12.5 ms                   |
| Subtalker (HF, 5 layers)     | ~2.0 ms     | ~25 ms                     |
| Codec decode (chunked)       | ~10 ms / 4f | ~32 ms                     |
| **Total per second of audio**| —           | **~70 ms** (RTF ≈ 0.07)    |

**TTFC budget**: prefill(50) + 4×(1+2)ms + codec(10) ≈ **72 ms**.

These are estimates — the kernel hot loop is the only stage we have published
numbers for (1,036 tok/s = 0.97 ms/step on RTX 5090 from
[the upstream blog](https://blog.alpindale.net/posts/5090_decode_optimization/)).
Real numbers go in benchmarks/ after the first hardware run.

---

## 9. What's not implemented / what needs hardware validation

| Item | Status | Notes |
|---|---|---|
| KV cache hand-off layout (HF → kernel) | Extractor confirmed CPU-side against synthetic DynamicCache (28 layers × 8 kv_heads × T × 128 → matches `MegakernelDecoder.set_kv_prefix` expectations). Real KV-tensor SEMANTICS still need hardware-side argmax-equality A/B vs one HF step. | `tts/pipeline.py: _extract_kv_for_kernel`. |
| MRoPE | Approximated with unified position counter. | Real fix requires patching kernel's RoPE application. For single-channel speech all three position dims collapse to the same advance, so theoretically a no-op. Verify on audio. |
| Subtalker hidden-state staleness | **Fixed.** Kernel exposes post-norm hidden via `g_normalized` buffer (= `self.norm_out`, float32 [1024]); `step_from_hidden` returns it as bf16. Subtalker now sees the *current* step's hidden. | `tts/talker.py: step_from_hidden`. |
| Codec streaming via internal sliding window | Currently re-decodes a 4-frame overlap per call. | `qwen_tts.core.tokenizer_12hz` doesn't expose a streaming API in 0.1.x; investigate v2. |
| Default-voice (no-ref-audio) synthesis | Currently raises in both backends. | Needs a default x-vector seed that `qwen_tts` doesn't expose directly; ship voice-clone path first. |
| Demo recording | Not produced — requires hardware. | |
| Benchmarks | Harness in place; numbers pending hardware run. | |

---

## 10. File map

```
Decoder/
├── DESIGN.md                          ← this file
├── README.md
├── requirements.txt
├── csrc/                              ← symlinked from AlpinDale/qwen_megakernel (unmodified)
│   ├── kernel.cu
│   └── torch_bindings.cpp
├── qwen_megakernel_tts/
│   ├── build.py                       ← JIT compile, flags match upstream verbatim
│   └── model.py                       ← TalkerKernelWeights: extract+pack codec-side weights
├── tts/
│   ├── talker.py                      ← MegakernelDecoder (single-step kernel wrapper)
│   ├── code_predictor.py              ← thin wrapper over loaded subtalker
│   ├── codec.py                       ← streaming wrapper over speech_tokenizer
│   └── pipeline.py                    ← orchestrator (hf-prefill → mega-decode → predictor → codec)
├── server/
│   └── app.py                         ← FastAPI WebSocket TTS server
├── pipecat_integration/
│   ├── tts_service.py                 ← MegakernelTTSService (subclasses pipecat TTSService)
│   └── demo.py                        ← STT → LLM → TTS voice loop
├── benchmarks/
│   ├── bench_ttfc.py
│   ├── bench_rtf.py
│   └── bench_e2e.py
└── scripts/
    ├── install.sh
    └── download_models.py
```
