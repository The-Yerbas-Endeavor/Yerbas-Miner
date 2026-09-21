#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-production-latency}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
LOG_DIR="$ROOT/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/dualhash-ilp-${STAMP}.log"
CACHE_ROOT="$ROOT/.bench-cache/dualhash-ilp-${STAMP}"

mkdir -p "$LOG_DIR" "$CACHE_ROOT"

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
echo " YERBAS TRUE DUAL-HASH ILP PHASE-2 A/B"
echo "============================================================"
echo " Binary:      $BIN"
echo " GPUs:        ${GPUS[*]}"
echo " Variants:    ${VARIANTS[*]}"
echo " Batch:       3584"
echo " Cache root:  $CACHE_ROOT"
echo " Production:  unchanged (experimental gate only)"
echo "============================================================"
} | tee "$LOG"

for gpu in "${GPUS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        CASE_CACHE="$CACHE_ROOT/gpu${gpu}-${variant}"
        mkdir -p "$CASE_CACHE"

        {
        echo
        echo "################################################################"
        echo "### GPU $gpu | CN-$variant"
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

            export XDG_CACHE_HOME="$CASE_CACHE"
            export YERBAS_CUDA_BENCH_RAW_BATCH=1
            export YERBAS_BENCH_FORCE_CN_VARIANT="$variant"
            export YERBAS_CUDA_RETUNE=1
            export YERBAS_CN_DUALHASH_EXPERIMENT=1

            "$BIN" "$gpu" 3584
        ) 2>&1 | tee -a "$LOG"
    done
done

echo | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo " RESULT INDEX" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
grep -E '### GPU|dualhash-parity|baseline-regs|dualhash-regs|candidate=(baseline|dualhash)|dualhash-median|production=|Best batch:' "$LOG" | tee -a "$LOG.index" || true

echo
echo "Full log:  $LOG"
echo "Index log: $LOG.index"
echo
echo "Promotion rule:"
echo "  parity PASS on both GPUs and both variants,"
echo "  then require a repeatable >=2% phase-2 win before any production trial."
