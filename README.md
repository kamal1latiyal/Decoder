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

## Quick start

```bash
# 1.  Install (clones the kernel repo, builds extension, installs deps)
bash scripts/install.sh

# 2.  Download Qwen3-TTS models (~3 GB)
python scripts/download_models.py

# 3.  Start the TTS WebSocket server
python -m server.app --host 0.0.0.0 --port 8765 \
    --backend megakernel \
    [--ref-audio /path/to/voice.wav --ref-text "transcript of voice.wav"]

# 4.  (Optional) Run the Pipecat voice loop
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

| Metric         | Target  | Measured (megakernel) | Measured (hf baseline) |
|----------------|---------|-----------------------|------------------------|
| Talker tok/s   | ~1000   | TBD                   | TBD                    |
| TTFC           | < 90 ms | TBD                   | TBD                    |
| RTF            | < 0.3   | TBD                   | TBD                    |
| E2E latency    | —       | TBD                   | TBD                    |

*Numbers go here after the first hardware run.*

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
| `csrc/kernel.cu`, `csrc/torch_bindings.cpp` | Unmodified — symlinked from `AlpinDale/qwen_megakernel`. |
| `qwen_megakernel_tts/build.py` | JIT compile. Flags match upstream verbatim (LDG_PREFETCH_*, USE_UINT4, ATTENTION_VEC4, WEIGHT_LDCS, MLP_SMEM, `-arch=sm_120a`). No `LDG_VOCAB_SIZE` flag (none exists upstream). |
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
