#!/usr/bin/env bash
# bootstrap.sh — single-command setup + smoke + benchmarks on a fresh RTX 5090 box.
#
# Run from the repo root:
#   bash scripts/bootstrap.sh
#
# What it does:
#   1. scripts/install.sh         (clone kernel, JIT compile, install deps)
#   2. scripts/download_models.py (pull HF models)
#   3. scripts/smoke_test.py      (bisect-style verification of every stage)
#   4. Start TTS server in background
#   5. Run TTFC / RTF / E2E benchmarks, save JSON results
#
# On any failure it exits non-zero with a clear stage label so you can
# bisect manually.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RESULTS_DIR="${REPO_DIR}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/bootstrap-$(date +%Y%m%dT%H%M%S).log"
SERVER_LOG="${RESULTS_DIR}/server.log"

bar() { printf "\n%s\n%s\n%s\n" "============================================================" " $1" "============================================================" | tee -a "${LOG_FILE}"; }
fail() { printf "\n!! STAGE FAILED: %s — see %s for details\n" "$1" "${LOG_FILE}" >&2; exit 1; }

bar "[0/5] Sanity: GPU + CUDA toolkit"
{
    nvidia-smi | head -3
    nvcc --version | tail -1
} 2>&1 | tee -a "${LOG_FILE}"
nvidia-smi | grep -q "RTX 5090" || echo "  WARNING: RTX 5090 not detected — kernel may not build."
nvcc --version | grep -qE "release (12\.8|12\.9|13\.)" || echo "  WARNING: CUDA toolkit < 12.8; build will likely fail."

bar "[1/5] install.sh (kernel + Python deps)"
bash scripts/install.sh 2>&1 | tee -a "${LOG_FILE}" || fail "install.sh"

# Re-activate venv for subsequent steps.
source "${REPO_DIR}/.venv/bin/activate"

bar "[2/5] Download Qwen3-TTS models"
python scripts/download_models.py 2>&1 | tee -a "${LOG_FILE}" || fail "download_models"

bar "[3/5] Smoke test (kernel → weights → decode → codec → PCM)"
python scripts/smoke_test.py 2>&1 | tee -a "${LOG_FILE}" || fail "smoke_test"

bar "[4/5] Start TTS server (background)"
nohup python -m server.app --host 0.0.0.0 --port 8765 --backend megakernel \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "  server PID: ${SERVER_PID}, log: ${SERVER_LOG}"

# Wait for /health to return ready (cold model load is ~30s).
echo -n "  waiting for /health... "
for i in $(seq 1 90); do
    if curl -sf http://localhost:8765/health 2>/dev/null | grep -q '"status":"ready"'; then
        echo "ready (after ${i}s)"
        break
    fi
    sleep 1
    [[ $i -eq 90 ]] && fail "server-did-not-start (see ${SERVER_LOG})"
done

trap "echo 'Stopping server pid=${SERVER_PID}'; kill ${SERVER_PID} 2>/dev/null || true" EXIT

bar "[5/5] Benchmarks"
python benchmarks/bench_ttfc.py --runs 5 2>&1 | tee "${RESULTS_DIR}/ttfc.txt"
python benchmarks/bench_rtf.py  --runs 3 2>&1 | tee "${RESULTS_DIR}/rtf.txt"
python benchmarks/bench_e2e.py        2>&1 | tee "${RESULTS_DIR}/e2e.txt"

bar "DONE"
echo "Results in: ${RESULTS_DIR}"
echo "Server log: ${SERVER_LOG}"
echo "Server still running (pid ${SERVER_PID}). Press Ctrl+C or run 'kill ${SERVER_PID}' to stop."
echo "Or: pkill -f 'server.app' to clean up."
