#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${YERBAS_PREFETCH_BENCH_BIN:-$ROOT/build-production-latency/cuda-batch-benchmark}"
BATCH="${YERBAS_PREFETCH_BATCH:-3584}"
GPUS="${YERBAS_PREFETCH_GPUS:-0 1}"
VARIANTS="${YERBAS_PREFETCH_VARIANTS:-lite turtle fast}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${YERBAS_PREFETCH_LOG_DIR:-$ROOT/logs/cn-prefetch-$STAMP}"
SUMMARY="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: missing benchmark binary: $BIN"
    echo "Build it first with:"
    echo '  cmake --build build-production-latency -j"$(nproc)" --target cuda-batch-benchmark'
    exit 1
fi

unset YERBAS_CUDA_RETUNE
unset YERBAS_GPU_AUTOTUNE
unset YERBAS_GPU_VARIANT_AUTOTUNE
unset YERBAS_CUDA_OVERLAP
unset YERBAS_DIAGNOSTICS

echo "============================================================"
echo " YERBAS CN LOCAL-TABLE-FIRST CANDIDATE MATRIX"
echo "============================================================"
echo " Binary:   $BIN"
echo " Batch:    $BATCH"
echo " GPUs:     $GPUS"
echo " Variants: $VARIANTS"
echo " Logs:     $LOG_DIR"
echo "============================================================"

: > "$SUMMARY"

for gpu in $GPUS; do
    for variant in $VARIANTS; do
        LOG="$LOG_DIR/gpu${gpu}-${variant}.log"
        echo
        echo "---- GPU $gpu | CN-$variant | batch $BATCH ----"
        YERBAS_BENCH_FORCE_CN_VARIANT="$variant"             stdbuf -oL -eL "$BIN" "$gpu" "$BATCH" 2>&1 | tee "$LOG"

        {
            echo "---- GPU $gpu | CN-$variant ----"
            grep -E               'CUDA CN production selector|CUDA CN production sample|production=4-lane-ttable-prefetch|prefetch-|Best batch|throughput:'               "$LOG" || true
            echo
        } >> "$SUMMARY"
    done
done

echo
echo "============================================================"
echo " CANDIDATE SUMMARY"
echo "============================================================"
cat "$SUMMARY"
echo
echo "Summary: $SUMMARY"
