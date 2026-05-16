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

# ── 3. Python environment ────────────────────────────────────────
echo ""
echo "[3/6] Setting up Python environment..."
VENV_DIR="${REPO_DIR}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created venv at ${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
echo "  Activated: $(which python)"

# ── 4. PyTorch (CUDA 12.8) ──────────────────────────────────────
echo ""
echo "[4/6] Installing PyTorch 2.7 with CUDA 12.8..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available : $(python -c 'import torch; print(torch.cuda.is_available())')"

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
