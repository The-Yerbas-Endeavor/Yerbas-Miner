#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-production-latency}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$ROOT/logs/bank8-fastlite-${STAMP}.log"
INDEX="$LOG.index"
CACHE_ROOT="$ROOT/.bench-cache/bank8-fastlite-${STAMP}"

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

if [[ -n "${YERBAS_BENCH_VARIANTS:-}" ]]; then
    read -r -a VARIANTS <<< "$YERBAS_BENCH_VARIANTS"
else
    VARIANTS=(fast lite)
fi

{
echo "============================================================"
echo " YERBAS BANK8 SHARED T-TABLE FAST/LITE A/B"
echo "============================================================"
echo " GPUs:       ${GPUS[*]}"
echo " Variants:   ${VARIANTS[*]}"
echo " Batch:      3584"
echo " Baseline:   4 KiB shared T-tables"
echo " Candidate:  32 KiB lane-classed 8x shared T-tables"
echo " Cache root: $CACHE_ROOT"
echo " Production: unchanged (experimental gate only)"
echo "============================================================"
} | tee "$LOG"

for gpu in "${GPUS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        case_cache="$CACHE_ROOT/gpu${gpu}-${variant}"
        mkdir -p "$case_cache"

        {
        echo
        echo "################################################################"
        echo "### GPU $gpu | CN-$variant | bank8"
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
            unset YERBAS_CN_DUALHASH_EXPERIMENT || true
            unset YERBAS_CN_DUALHASH_ONLY || true
            unset YERBAS_CN_DUALHASH_THREADS || true
            unset YERBAS_CN_SPARSE_WARP_HASHES || true
            unset YERBAS_CN_SPARSE_WARP_ONLY || true
            unset YERBAS_CN_MUL4_EXPERIMENT || true
            unset YERBAS_CN_PAIRLOAD_EXPERIMENT || true
            unset YERBAS_CN_2LANE_TTABLE_EXPERIMENT || true
            unset YERBAS_CN_READONLY_TTABLE_EXPERIMENT || true

            export XDG_CACHE_HOME="$case_cache"
            export YERBAS_CUDA_BENCH_RAW_BATCH=1
            export YERBAS_BENCH_FORCE_CN_VARIANT="$variant"
            export YERBAS_BENCH_WARMUP_SCANS=2
            export YERBAS_CUDA_RETUNE=1
            export YERBAS_CN_BANK8_EXPERIMENT=1

            "$BIN" "$gpu" 3584
        ) 2>&1 | tee -a "$LOG"
    done
done

{
echo "============================================================"
echo " RESULT INDEX"
echo "============================================================"
grep -E '^### GPU|experimental=4-lane-ttable-bank8|candidate=4-lane-ttable-bank8|baseline-median=.*experimental=' "$LOG" || true
} | tee "$INDEX"

echo
echo "Full log:  $LOG"
echo "Index log: $INDEX"
echo
echo "Promotion gate:"
echo "  parity PASS on both GPUs;"
echo "  zero local-memory spill;"
echo "  candidate shared memory ~= 32768 bytes;"
echo "  >=2% repeatable phase-2 win on the same variant;"
echo "  then repeat before any production trial."
