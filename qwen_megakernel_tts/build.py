"""
Builds the qwen_megakernel CUDA extension.

DEVIATION FROM UPSTREAM
-----------------------
Upstream kernel.cu hardcodes `constexpr int LDG_VOCAB_SIZE = 151936;` for
Qwen3-0.6B's text vocab. The Qwen3-TTS talker's codec vocab is 3072, which
means the LM-head kernel would do an OOB read of 148,864 rows past the
codec_head tensor → garbage argmax.

We patch upstream via install.sh (idempotent perl edit) to wrap the constexpr
in `#ifndef LDG_VOCAB_SIZE`, then pass `-DLDG_VOCAB_SIZE=3072` here. The
default (no -D, e.g. for re-using this build flow with Qwen3-0.6B) remains
151,936.

Other than that one constant, every kernel flag matches upstream verbatim —
see https://github.com/AlpinDale/qwen_megakernel/blob/master/qwen_megakernel/build.py
"""

import os
import torch.utils.cpp_extension

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSRC = os.path.join(_REPO_ROOT, "csrc")

_ext = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


KERNEL_FLAGS = [
    # Qwen3-TTS talker codec vocab = 3072.  Upstream kernel hardcodes
    # LDG_VOCAB_SIZE=151936 (Qwen3-0.6B text vocab); install.sh patches the
    # constexpr into an #ifndef so this -D wins.  Without this override the
    # LM-head kernel does an out-of-bounds read of 148,864 rows past the
    # codec_head tensor → garbage argmax.
    f"-DLDG_VOCAB_SIZE={_env_int('LDG_VOCAB_SIZE', 3072)}",
    f"-DLDG_NUM_BLOCKS={_env_int('LDG_NUM_BLOCKS', 128)}",
    f"-DLDG_BLOCK_SIZE={_env_int('LDG_BLOCK_SIZE', 512)}",
    f"-DLDG_LM_NUM_BLOCKS={_env_int('LDG_LM_NUM_BLOCKS', 1280)}",
    f"-DLDG_LM_BLOCK_SIZE={_env_int('LDG_LM_BLOCK_SIZE', 384)}",
    f"-DLDG_LM_ROWS_PER_WARP={_env_int('LDG_LM_ROWS_PER_WARP', 2)}",
    f"-DLDG_ATTN_BLOCKS={_env_int('LDG_ATTN_BLOCKS', 8)}",
    f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
    f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
    f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
    f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
    f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
    f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
    f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
    "-DLDG_USE_UINT4",
    "-DLDG_ATTENTION_VEC4",
    "-DLDG_WEIGHT_LDCS",
    "-DLDG_MLP_SMEM",
]

CUDA_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-arch=sm_120a",          # Blackwell / RTX 5090 (note: 'a' suffix is required)
    f"-I{_CSRC}",
] + KERNEL_FLAGS


def get_extension():
    """JIT-build (or return cached) the kernel. Registers torch.ops.qwen_megakernel_C.*"""
    global _ext
    if _ext is not None:
        return _ext

    sources = [
        os.path.join(_CSRC, "torch_bindings.cpp"),
        os.path.join(_CSRC, "kernel.cu"),
    ]
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Missing kernel source: {src}\n"
                "Run: bash scripts/install.sh  (clones AlpinDale/qwen_megakernel)"
            )

    _ext = torch.utils.cpp_extension.load(
        name="qwen_megakernel_C",     # must match upstream so torch.ops namespace matches
        sources=sources,
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=True,
    )
    return _ext
