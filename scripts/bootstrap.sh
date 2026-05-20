#!/usr/bin/env bash
# bootstrap.sh — single-command setup + smoke + benchmarks on a fresh RTX 5090 box.
#
# Run from the repo root:
#   bash scripts/bootstrap.sh
#
# Optional overrides (env vars):
#   REF_AUDIO   Path to a reference voice WAV (3-10s recommended).
#               If unset, auto-downloads a public-domain LJSpeech sample.
#   REF_TEXT    Transcript of REF_AUDIO. If unset and the default sample is
#               used, the known LJSpeech transcript is used.
#
# What it does:
#   1. Sanity: nvidia-smi, nvcc, RTX 5090 detection.
#   2. install.sh           (clone kernel, install Python deps, JIT compile)
#   3. download_models.py   (pull Qwen3-TTS HF weights ~3 GB)
#   4. ref-audio fetch      (if REF_AUDIO not set; ~420 KB public-domain WAV)
#   5. inspect_model.py     (CUDA-side API sanity check — fail-fast, ~1 min)
#   6. smoke_test.py        (9-stage bisect verification — fail-fast, ~5 min)
#   7. Start TTS server in background
#   8. Run TTFC / RTF / E2E benchmarks, save JSON results
#
# On any failure it exits non-zero with a clear stage label so you can
# bisect manually.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RESULTS_DIR="${REPO_DIR}/benchmarks/results"
REFS_DIR="${REPO_DIR}/refs"
mkdir -p "${RESULTS_DIR}" "${REFS_DIR}"

LOG_FILE="${RESULTS_DIR}/bootstrap-$(date +%Y%m%dT%H%M%S).log"
SERVER_LOG="${RESULTS_DIR}/server.log"

# Public-domain LJSpeech sample (16-bit mono PCM @ 22.05 kHz, ~416 KB).
# qwen_tts's wrapper handles arbitrary input sample rates internally.
DEFAULT_REF_URL="https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav"
DEFAULT_REF_TEXT="Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"

bar() { printf "\n%s\n%s\n%s\n" "============================================================" " $1" "============================================================" | tee -a "${LOG_FILE}"; }
fail() { printf "\n!! STAGE FAILED: %s — see %s for details\n" "$1" "${LOG_FILE}" >&2; exit 1; }

bar "[0/8] Sanity: GPU + CUDA toolkit"
{
    nvidia-smi | head -3 || true
    nvcc --version | tail -1 || true
} 2>&1 | tee -a "${LOG_FILE}"
nvidia-smi 2>/dev/null | grep -q "RTX 5090" || echo "  WARNING: RTX 5090 not detected — kernel may not build (requires sm_120a)."
nvcc --version 2>/dev/null | grep -qE "release (12\.8|12\.9|13\.)" || echo "  WARNING: CUDA toolkit < 12.8; build will likely fail."

bar "[1/8] install.sh (kernel + Python deps)"
bash scripts/install.sh 2>&1 | tee -a "${LOG_FILE}" || fail "install.sh"

# Re-activate venv for subsequent steps.
source "${REPO_DIR}/.venv/bin/activate"

bar "[2/8] Download Qwen3-TTS models"
python scripts/download_models.py 2>&1 | tee -a "${LOG_FILE}" || fail "download_models"

bar "[3/8] Reference voice"
# Resolve REF_AUDIO: explicit env var > refs/voice.wav (user-supplied or auto-download).
if [[ -n "${REF_AUDIO:-}" ]]; then
    if [[ ! -f "${REF_AUDIO}" ]]; then
        echo "ERROR: REF_AUDIO=${REF_AUDIO} not found" >&2
        fail "ref-audio-missing"
    fi
    echo "  using user-provided REF_AUDIO=${REF_AUDIO}"
else
    REF_AUDIO="${REFS_DIR}/voice.wav"
    if [[ -f "${REF_AUDIO}" ]]; then
        echo "  found existing ${REF_AUDIO} ($(du -h "${REF_AUDIO}" | cut -f1)) — using as-is"
    else
        echo "  fetching default public-domain sample (LJSpeech LJ001-0001)..."
        curl -fsSL "${DEFAULT_REF_URL}" -o "${REF_AUDIO}" \
            || fail "ref-audio-download (URL: ${DEFAULT_REF_URL})"
        # Default sample comes with a known transcript — use it iff REF_TEXT unset.
        REF_TEXT="${REF_TEXT:-${DEFAULT_REF_TEXT}}"
    fi
fi
# REF_TEXT is optional. If set, qwen_tts uses ICL mode (slightly higher cloning
# fidelity). If unset, it falls back to x-vector-only mode (speaker embedding
# only — works fine for cloning voice characteristics, doesn't need transcript).
if [[ -n "${REF_TEXT:-}" ]]; then
    echo "  REF_TEXT set (${#REF_TEXT} chars) — using ICL voice-clone mode"
else
    echo "  REF_TEXT not set — using x-vector-only mode (no transcript needed)"
fi
export REF_AUDIO
[[ -n "${REF_TEXT:-}" ]] && export REF_TEXT

bar "[4/8] inspect_model.py (CUDA, ~1 min) — static API check"
python scripts/inspect_model.py --device cuda 2>&1 | tee -a "${LOG_FILE}" || fail "inspect_model"

bar "[5/8] Smoke test — 9-stage verification (CUDA kernel + KV parity + E2E)"
python scripts/smoke_test.py 2>&1 | tee -a "${LOG_FILE}" || fail "smoke_test"

bar "[6/8] Start TTS server (background)"
SERVER_ARGS=(--host 0.0.0.0 --port 8765 --backend megakernel --ref-audio "${REF_AUDIO}")
[[ -n "${REF_TEXT:-}" ]] && SERVER_ARGS+=(--ref-text "${REF_TEXT}")
nohup python -m server.app "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
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

bar "[7/8] Benchmarks"
python benchmarks/bench_ttfc.py --runs 5 2>&1 | tee "${RESULTS_DIR}/ttfc.txt"
python benchmarks/bench_rtf.py  --runs 3 2>&1 | tee "${RESULTS_DIR}/rtf.txt"
python benchmarks/bench_e2e.py        2>&1 | tee "${RESULTS_DIR}/e2e.txt"

bar "[8/8] DONE"
echo "Results in: ${RESULTS_DIR}"
echo "Server log: ${SERVER_LOG}"
echo "Server still running (pid ${SERVER_PID})."
echo "  - WebSocket TTS:  ws://localhost:8765/synthesize"
echo "  - Health         : http://localhost:8765/health"
echo "  - Metrics        : http://localhost:8765/metrics"
echo "Press Ctrl+C or run 'kill ${SERVER_PID}' to stop. Or: pkill -f 'server.app'."
