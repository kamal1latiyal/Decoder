"""
Download Qwen3-TTS models from Hugging Face Hub.

Downloads:
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base   (~2.5 GB)  talker + code predictor
  - Qwen/Qwen3-TTS-Tokenizer-12Hz    (~0.5 GB)  codec decoder

Run:
  python scripts/download_models.py [--cache-dir ~/.cache/huggingface]
"""

import argparse
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

MODELS = [
    {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "description": "Talker decoder + code predictor (~2.5 GB)",
    },
    {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "description": "Codec decoder (~0.5 GB)",
    },
]


def download_all(cache_dir: str | None = None):
    for model in MODELS:
        print(f"\nDownloading {model['repo_id']} — {model['description']}")
        try:
            path = snapshot_download(
                repo_id=model["repo_id"],
                cache_dir=cache_dir,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            print(f"  ✓ Saved to: {path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print(f"    Try: huggingface-cli login  (if model requires authentication)")
            sys.exit(1)

    print("\nAll models downloaded.")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-TTS models")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    args = parser.parse_args()
    download_all(args.cache_dir)


if __name__ == "__main__":
    main()
