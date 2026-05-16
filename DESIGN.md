# System Design — RTX 5090 Megakernel → Qwen3-TTS on Pipecat

## 1. Problem

Use AlpinDale's [`qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel)
(~1,000 tok/s for Qwen3-0.6B on RTX 5090) as the autoregressive-decode backend
for Qwen3-TTS's talker, exposed as a streaming TTS service inside a Pipecat
voice pipeline.

---

## 2. Architecture match — what's actually compatible

The first draft of this design (commit history) claimed the megakernel was a
near drop-in for the talker. Reading
[`config.json`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/config.json)
on 2026-05-16 shows it isn't. Honest comparison:

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

Two consequences:

1. **The megakernel cannot do text prefill.** The talker projects text
   embeddings from `text_hidden_size=2048 → hidden_size=1024` before entering
   the transformer stack — the kernel has no such projection. We do prefill
   in HF and hand the warm KV cache to the kernel.

2. **The codec decode loop *is* a Qwen3-0.6B-shaped transformer.** Once the
   KV cache is warm and we're feeding codec tokens (input embed = codec embed
   [3072,1024]), the per-step decode matches the megakernel exactly. No kernel
   modification needed. The vocab is implicit in the `embed_weight` /
   `lm_head_weight` tensor shapes — there is no `LDG_VOCAB_SIZE` flag in
   upstream and we don't add one.

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
- `Qwen3TTSModel.generate(..., max_new_tokens=1)` runs the talker over the
  text prompt with speaker conditioning, produces past_key_values + the first
  codec token + the last hidden state.
- Cost is ~30–80 ms for a typical 30–60 text-token prompt. Amortised over
  the whole utterance, this is the only stage that doesn't benefit from the
  kernel.

### Stage 2 — megakernel codec decode (hot loop, 12.5 Hz of audio)
- `MegakernelDecoder.step(token)` calls
  `torch.ops.qwen_megakernel_C.decode(...)` with the codec-vocab embed +
  lm_head and the warmed KV cache. Returns next codec token id.
- Target: ≈1 ms / token. 1 s of audio = 12.5 tokens = ~12.5 ms.

### Stage 3 — code predictor (HF, 5 layers, ~2 ms/token)
- The official subtalker is called per group-0 token to autoregressively emit
  groups 1..15 within the frame. Not a throughput bottleneck.

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
`bmax_vals/idxs` are sized 4096 (upstream's worst-case bound for the LM head
block-max reduction), not 1280. The first draft used 1280, which would
silently corrupt the argmax for the talker's smaller codec vocab — a latent
bug; the kernel computes block indices assuming 4096 entries.

### 4.4 Speaker conditioning
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
| KV cache hand-off layout (HF → kernel) | Coded against documented HF layout `[batch, kv_heads, seq, head_dim]`; **needs hardware run to confirm** transposition matches kernel's `[layer, kv_heads, seq, head_dim]`. | `MegakernelDecoder.set_kv_prefix()` is the single edit point. |
| MRoPE | Approximated with unified position counter. | Real fix requires patching kernel's RoPE application. |
| Subtalker hidden-state staleness | Uses talker's pre-step hidden state for the next step's predictor (kernel doesn't expose hidden). | Predictor is robust to one-step-stale conditioning; verify on audio. |
| Codec streaming via internal sliding window | Currently re-decodes a 4-frame overlap per call. | qwen_tts.core.tokenizer_12hz may expose a streaming API; investigate. |
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
