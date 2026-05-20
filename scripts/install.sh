#!/usr/bin/env bash
# install.sh — Full setup for Qwen3-TTS Megakernel project
# Run on: Ubuntu 22.04+ with CUDA 12.8, RTX 5090
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEGAKERNEL_REPO="https://github.com/AlpinDale/qwen_megakernel"
MEGAKERNEL_DIR="${REPO_DIR}/.megakernel_upstream"

echo "============================================================"
echo " Qwen3-TTS Megakernel — Install Script"
echo " Working directory: ${REPO_DIR}"
echo "============================================================"

# ── 1. Check CUDA ────────────────────────────────────────────────
echo ""
echo "[1/6] Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA Toolkit 12.8."
    exit 1
fi
CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
echo "  CUDA version: ${CUDA_VER}"
if ! nvidia-smi | grep -q "RTX 5090"; then
    echo "  WARNING: RTX 5090 not detected. Kernel requires sm_120 (Blackwell)."
fi

# ── 2. Clone upstream qwen_megakernel for csrc/ ──────────────────
echo ""
echo "[2/6] Fetching qwen_megakernel CUDA sources..."
if [ ! -d "${MEGAKERNEL_DIR}" ]; then
    git clone --depth 1 "${MEGAKERNEL_REPO}" "${MEGAKERNEL_DIR}"
    echo "  Cloned to ${MEGAKERNEL_DIR}"
else
    echo "  Already exists at ${MEGAKERNEL_DIR}, pulling latest..."
    git -C "${MEGAKERNEL_DIR}" pull --ff-only
fi

# Symlink csrc/ into this repo (kernel source is used unmodified)
if [ ! -e "${REPO_DIR}/csrc/kernel.cu" ]; then
    rm -rf "${REPO_DIR}/csrc"
    ln -s "${MEGAKERNEL_DIR}/csrc" "${REPO_DIR}/csrc"
    echo "  Symlinked csrc/ → ${MEGAKERNEL_DIR}/csrc"
else
    echo "  csrc/ already linked."
fi

# ── 2b. Patch upstream kernel.cu so LDG_VOCAB_SIZE is overridable ─────────
# Upstream hardcodes LDG_VOCAB_SIZE=151936 (Qwen3-0.6B text vocab). The TTS
# talker's codec vocab is 3072, so we MUST override it; otherwise the LM-head
# kernel does an OOB read of 148,864 rows past the codec_head tensor.
#
# The patch wraps the constexpr in #ifndef so -DLDG_VOCAB_SIZE=N from the
# build flags wins. Idempotent: the second run finds the original line gone
# and does nothing.
KCU="${MEGAKERNEL_DIR}/csrc/kernel.cu"
if grep -q "^constexpr int LDG_VOCAB_SIZE = 151936;" "${KCU}"; then
    echo "  Patching kernel.cu to allow LDG_VOCAB_SIZE override..."
    # Use perl instead of sed for portable in-place multi-line replace.
    perl -i -pe 's{^constexpr int LDG_VOCAB_SIZE = 151936;$}{#ifndef LDG_VOCAB_SIZE\n#define LDG_VOCAB_SIZE 151936\n#endif}' "${KCU}"
    grep -q "^#ifndef LDG_VOCAB_SIZE" "${KCU}" || { echo "  PATCH FAILED"; exit 1; }
    echo "  Kernel patched: LDG_VOCAB_SIZE is now overridable from build flags."
elif grep -q "^#ifndef LDG_VOCAB_SIZE" "${KCU}"; then
    echo "  kernel.cu already patched (idempotent)."
else
    echo "  WARNING: kernel.cu doesn't match expected upstream layout — vocab patch skipped."
    echo "  Upstream may have changed; verify LDG_VOCAB_SIZE handling manually."
fi

# ── 3. Python environment ────────────────────────────────────────
echo ""
echo "[3/6] Setting up Python environment..."
VENV_DIR="${REPO_DIR}/.venv"
# --system-site-packages so NGC / vast.ai images' pre-installed PyTorch is
# visible inside the venv. Without this, we'd shadow NGC's CUDA-13-optimised
# PyTorch with a cu128 wheel — works but wastes ~3 min and the optimisations.
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "  Created venv (with --system-site-packages) at ${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
echo "  Activated: $(which python)"

# ── 4. PyTorch ──────────────────────────────────────────────────
echo ""
echo "[4/6] Checking / installing PyTorch (need >= 2.7 with CUDA)..."
pip install --upgrade pip --quiet

# Detect whether a usable PyTorch is already present.  NGC / Vast templates
# usually ship a CUDA-matched PyTorch 2.7+; if so we use it as-is.  Otherwise
# we install from the cu128 wheel index (compatible with CUDA 12.8 ≤ X < 14).
if python -c "
import sys
try:
    import torch
    v = torch.__version__.split('+')[0].split('a')[0].split('.dev')[0]
    major, minor = [int(x) for x in v.split('.')[:2]]
    ok = (major, minor) >= (2, 7) and torch.cuda.is_available()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    echo "  ✓ Existing PyTorch is usable: $(python -c 'import torch; print(torch.__version__)')"
    echo "    CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
else
    echo "  No usable PyTorch found — installing cu128 wheels..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
    echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "  CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"
fi

# ── 5. Python dependencies ──────────────────────────────────────
echo ""
echo "[5/6] Installing Python dependencies..."
pip install ninja --quiet                          # JIT compilation
pip install -r "${REPO_DIR}/requirements.txt" --quiet
echo "  Dependencies installed."

# ── 6. JIT compile megakernel (pre-warm) ────────────────────────
echo ""
echo "[6/6] Pre-compiling megakernel with TTS constants..."
echo "  (This may take 60–120s on first run...)"
python -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
from qwen_megakernel_tts.build import get_extension
ext = get_extension()
print('  Kernel compiled successfully:', ext)
"

echo ""
echo "============================================================"
echo " Installation complete!"
echo ""
echo " Next steps:"
echo "   source ${VENV_DIR}/bin/activate"
echo "   python scripts/download_models.py"
echo "   python -m server.app --port 8765"
echo "============================================================"
