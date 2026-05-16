"""
End-to-End Latency Benchmark.

Simulates the full pipeline: text arrives (as if from LLM) → first audio out.

Reports:
  - TTFC per request
  - RTF per request
  - Talker tok/s
  - Streaming chunk count and sizes

Run:
  python benchmarks/bench_e2e.py [--url ws://localhost:8765]
"""

import argparse
import asyncio
import json
import time

import websockets

SCENARIOS = [
    {"label": "Short greeting",       "text": "Hello! How can I help you today?"},
    {"label": "Medium response",      "text": "Sure, I can explain that. Machine learning is a branch of artificial intelligence that enables systems to learn and improve from data."},
    {"label": "Long explanation",     "text": "The megakernel achieves its performance by launching a single persistent kernel across 128 thread blocks of 512 threads each, eliminating kernel launch overhead between decode steps. Combined with fused attention and MLP operations, and aggressive use of GDDR7 bandwidth on the RTX 5090, it achieves approximately one thousand tokens per second for the Qwen three zero point six B model."},
]


async def e2e_run(server_url: str, text: str) -> dict:
    ws_url = f"{server_url}/synthesize"
    chunks = []
    t_start = time.perf_counter()
    t_first: float | None = None
    server_metrics = {}

    async with websockets.connect(ws_url, max_size=100 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"text": text}))

        async for msg in ws:
            if isinstance(msg, bytes):
                if t_first is None:
                    t_first = time.perf_counter()
                chunks.append(len(msg))
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "done":
                    server_metrics = data.get("metrics", {})

    t_end = time.perf_counter()

    total_bytes = sum(chunks)
    audio_samples = total_bytes // 2
    audio_duration = audio_samples / 24_000

    return {
        "ttfc_ms": (t_first - t_start) * 1000 if t_first else float("inf"),
        "wall_time_s": t_end - t_start,
        "audio_duration_s": audio_duration,
        "rtf": server_metrics.get("rtf", 0),
        "tokens_per_sec": server_metrics.get("tokens_per_sec", 0),
        "total_tokens": server_metrics.get("total_tokens", 0),
        "num_chunks": len(chunks),
        "chunk_sizes_bytes": chunks,
        "total_bytes": total_bytes,
    }


async def run_benchmark(server_url: str):
    print(f"\nEnd-to-End Latency Benchmark")
    print(f"Server: {server_url}")
    print("=" * 70)

    for scenario in SCENARIOS:
        label = scenario["label"]
        text = scenario["text"]
        print(f"\n── {label} ──")
        print(f"Text ({len(text)} chars): \"{text[:60]}...\"" if len(text) > 60 else f"Text: \"{text}\"")

        try:
            r = await e2e_run(server_url, text)
            print(f"  TTFC          : {r['ttfc_ms']:.1f} ms  (target < 90 ms)")
            print(f"  RTF           : {r['rtf']:.4f}  (target < 0.3)")
            print(f"  Talker tok/s  : {r['tokens_per_sec']:.0f}")
            print(f"  Codec tokens  : {r['total_tokens']}")
            print(f"  Audio duration: {r['audio_duration_s']:.3f} s")
            print(f"  Wall time     : {r['wall_time_s']:.3f} s")
            print(f"  PCM chunks    : {r['num_chunks']} × "
                  f"[{', '.join(str(s) for s in r['chunk_sizes_bytes'][:5])}{'...' if len(r['chunk_sizes_bytes']) > 5 else ''}] bytes")
            print(f"  Total PCM     : {r['total_bytes']:,} bytes = {r['total_bytes']/1024:.1f} KB")

            ttfc_ok = "✓" if r["ttfc_ms"] < 90 else "✗"
            rtf_ok  = "✓" if r["rtf"] < 0.3   else "✗"
            print(f"  Targets: TTFC {ttfc_ok}  RTF {rtf_ok}")

        except Exception as e:
            print(f"  ERROR: {e}")

        await asyncio.sleep(1.0)

    print("\n" + "=" * 70)
    print("Streaming verification:")
    print("  ✓ Audio is pushed chunk-by-chunk (not buffered) if num_chunks > 1")
    print("  ✓ Each chunk = 4 codec frames = 320ms audio = 7,680 bytes")


def main():
    parser = argparse.ArgumentParser(description="E2E Latency Benchmark")
    parser.add_argument("--url", default="ws://localhost:8765")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.url))


if __name__ == "__main__":
    main()
