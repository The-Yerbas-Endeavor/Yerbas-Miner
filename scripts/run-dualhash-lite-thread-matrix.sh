#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-production-latency}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$ROOT/logs/dualhash-lite-thread-matrix-${STAMP}.log"
CACHE_ROOT="$ROOT/.bench-cache/dualhash-lite-thread-matrix-${STAMP}"

mkdir -p "$ROOT/logs" "$CACHE_ROOT"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building cuda-batch-benchmark..."
cmake --build "$BUILD_DIR" --target cuda-batch-benchmark -j"$(nproc)"

if [[ -n "${YERBAS_BENCH_GPUS:-}" ]]; then
    read -r -a GPUS <<< "$YERBAS_BENCH_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
else
    GPUS=(0)
fi

THREADS=(128 160 192 224 256 288 320 352 384 448 512 576)

{
echo "============================================================"
echo " YERBAS CN-LITE DUAL-HASH THREAD MATRIX"
echo "============================================================"
echo " GPUs:        ${GPUS[*]}"
echo " Batch:       3584"
echo " Threads:     ${THREADS[*]}"
echo " Variant:     CN-Lite"
echo " Candidates:  baseline vs dual-hash only"
echo " Cache root:  $CACHE_ROOT"
echo "============================================================"
} | tee "$LOG"

for gpu in "${GPUS[@]}"; do
    for threads in "${THREADS[@]}"; do
        blocks=$(( (7168 + threads - 1) / threads ))
        cache="$CACHE_ROOT/gpu${gpu}-t${threads}"
        mkdir -p "$cache"

        {
        echo
        echo "################################################################"
        echo "### GPU $gpu | CN-Lite | dualhash-threads=$threads | blocks=$blocks"
        echo "################################################################"
        } | tee -a "$LOG"

        (
            unset YERBAS_CN_PHASE_RETUNE || true
            unset YERBAS_CN_PHASE_THREADS || true
            unset YERBAS_CN_SETUP_THREADS || true
            unset YERBAS_CN_FINAL_THREADS || true
            unset YERBAS_DIAGNOSTICS || true
            unset YERBAS_CUDA_PROFILE || true
            unset YERBAS_CUDA_OVERLAP || true
            unset YERBAS_GPU_AUTOTUNE || true
            unset YERBAS_GPU_VARIANT_AUTOTUNE || true
            unset YERBAS_CN_MUL4_EXPERIMENT || true
            unset YERBAS_CN_PAIRLOAD_EXPERIMENT || true
            unset YERBAS_CN_2LANE_TTABLE_EXPERIMENT || true
            unset YERBAS_CN_READONLY_TTABLE_EXPERIMENT || true

            export XDG_CACHE_HOME="$cache"
            export YERBAS_CUDA_BENCH_RAW_BATCH=1
            export YERBAS_BENCH_FORCE_CN_VARIANT=lite
            export YERBAS_BENCH_WARMUP_SCANS=2
            export YERBAS_CUDA_RETUNE=1
            export YERBAS_CN_DUALHASH_EXPERIMENT=1
            export YERBAS_CN_DUALHASH_ONLY=1
            export YERBAS_CN_DUALHASH_THREADS="$threads"

            "$BIN" "$gpu" 3584
        ) 2>&1 | tee -a "$LOG"
    done
done

INDEX="$LOG.index"
{
echo "============================================================"
echo " RESULT INDEX"
echo "============================================================"
grep -E '^### GPU|dualhash-parity|baseline-median=.*dualhash-median' "$LOG" || true
} | tee "$INDEX"

echo
echo "Full log:  $LOG"
echo "Index log: $INDEX"
echo
echo "Interpretation:"
echo "  256 threads = exactly 28 blocks for 3584 dual-hash pairs -> one block per SM."
echo "  128 threads = exactly 56 blocks -> two blocks per SM."
echo "  Compare repeatability across both GTX 1080 Ti cards before promoting anything."
