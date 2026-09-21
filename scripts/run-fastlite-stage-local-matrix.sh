#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-production-latency}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# First-pass structural test for the production bottleneck seen in the Sept. 20-21
# overnight run. Keep the known-good 3584 CN chunk and vary only the outer
# GhostRider batch. This tests whether the 15 conventional stages can profit from
# a larger outer batch while memory-heavy CN stages remain inside the existing
# scratchpad budget.
CN_CHUNK="${YERBAS_FASTLITE_CN_CHUNK:-3584}"
CANDIDATES=(3584 4480 5376 6272 7168)
TRIPLES=(
    "dark,fast,lite"
    "darklite,fast,lite"
    "fast,lite,turtle"
    "fast,lite,turtlelite"
)

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Configuring CUDA build: $BUILD_DIR"
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building cuda-batch-benchmark..."
cmake --build "$BUILD_DIR" --target cuda-batch-benchmark -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: benchmark binary was not produced: $BIN" >&2
    exit 1
fi

if [[ -n "${YERBAS_BENCH_GPUS:-}" ]]; then
    read -r -a GPUS <<< "$YERBAS_BENCH_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
else
    GPUS=(0)
fi

if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "ERROR: no GPU ids selected" >&2
    exit 1
fi

LOG="$LOG_DIR/fastlite-stage-local-matrix-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "============================================================"
echo " YERBAS FAST+LITE STAGE-LOCAL BATCH MATRIX"
echo "============================================================"
echo " Binary:       $BIN"
echo " GPUs:         ${GPUS[*]}"
echo " CN chunk:     $CN_CHUNK"
echo " Outer sizes:  ${CANDIDATES[*]}"
echo " Retuning:     OFF"
echo " Diagnostics:  OFF"
echo " Log:          $LOG"
echo
echo "Purpose:"
echo "  Baseline = one full 3584 batch through all 18 stages."
echo "  Candidate = larger outer batch; CN stages chunked at the"
echo "              existing 3584 scratchpad budget."
echo "============================================================"

for gpu in "${GPUS[@]}"; do
    for triple in "${TRIPLES[@]}"; do
        echo
        echo "################################################################"
        echo "### GPU $gpu | CN triple $triple"
        echo "################################################################"

        echo
        echo "--- BASELINE: full common batch 3584 ---"
        env \
            -u YERBAS_CUDA_BENCH_STAGE_LOCAL \
            -u YERBAS_CUDA_BENCH_CN_CHUNK \
            -u YERBAS_CN_PHASE_RETUNE \
            -u YERBAS_CN_PHASE_THREADS \
            -u YERBAS_CN_SETUP_THREADS \
            -u YERBAS_CN_FINAL_THREADS \
            -u YERBAS_DIAGNOSTICS \
            -u YERBAS_CUDA_PROFILE \
            -u YERBAS_CUDA_RETUNE \
            -u YERBAS_CUDA_OVERLAP \
            -u YERBAS_GPU_AUTOTUNE \
            -u YERBAS_GPU_VARIANT_AUTOTUNE \
            YERBAS_CUDA_BENCH_RAW_BATCH=1 \
            YERBAS_BENCH_FORCE_CN_TRIPLE="$triple" \
            "$BIN" "$gpu" 3584

        echo
        echo "--- CANDIDATE: stage-local CN chunk=$CN_CHUNK ---"
        env \
            -u YERBAS_CUDA_BENCH_RAW_BATCH \
            -u YERBAS_CN_PHASE_RETUNE \
            -u YERBAS_CN_PHASE_THREADS \
            -u YERBAS_CN_SETUP_THREADS \
            -u YERBAS_CN_FINAL_THREADS \
            -u YERBAS_DIAGNOSTICS \
            -u YERBAS_CUDA_PROFILE \
            -u YERBAS_CUDA_RETUNE \
            -u YERBAS_CUDA_OVERLAP \
            -u YERBAS_GPU_AUTOTUNE \
            -u YERBAS_GPU_VARIANT_AUTOTUNE \
            YERBAS_CUDA_BENCH_STAGE_LOCAL=1 \
            YERBAS_CUDA_BENCH_CN_CHUNK="$CN_CHUNK" \
            YERBAS_BENCH_FORCE_CN_TRIPLE="$triple" \
            "$BIN" "$gpu" "${CANDIDATES[@]}"
    done
done

echo
echo "============================================================"
echo " MATRIX COMPLETE"
echo "============================================================"
echo "Log: $LOG"
echo
echo "Compact result index:"
grep -E '^(### GPU|--- BASELINE|--- CANDIDATE|Best batch:)' "$LOG" || true
echo
echo "Do not promote a winner from one GPU/triple alone."
echo "Send the complete log for cross-GPU/triple comparison."
