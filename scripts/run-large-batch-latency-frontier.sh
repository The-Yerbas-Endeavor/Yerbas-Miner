#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-stale-aware}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

CANDIDATES=(3584 4480 5376 6272 7168 8960 10752 11648 13440 16128 17920)
TRIPLES=(
    "dark,darklite,turtle"
    "dark,darklite,turtlelite"
)

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building cuda-batch-benchmark..."
cmake --build "$BUILD_DIR" --target cuda-batch-benchmark -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: benchmark binary not found: $BIN" >&2
    exit 1
fi

if [[ -n "${YERBAS_BENCH_GPUS:-}" ]]; then
    read -r -a GPUS <<< "$YERBAS_BENCH_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
else
    GPUS=(0)
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/large-batch-latency-frontier-${STAMP}.log"

{
echo "============================================================"
echo " YERBAS LARGE-BATCH THROUGHPUT / LATENCY FRONTIER"
echo "============================================================"
echo " GPUs:       ${GPUS[*]}"
echo " Triples:    ${TRIPLES[*]}"
echo " Candidates: ${CANDIDATES[*]}"
echo " Passes:     2 (ascending then descending)"
echo " Setup/final geometry: forced 32/128 for fair comparison"
echo " Metric:     raw full-pipeline H/s and scan latency"
echo " Production: unchanged"
echo " Log:        $LOG"
echo "============================================================"
} | tee "$LOG"

run_case() {
    local gpu="$1"
    local triple="$2"
    local pass="$3"
    shift 3
    local sizes=("$@")

    {
        echo
        echo "################################################################"
        echo "### GPU $gpu | CN triple $triple | pass=$pass"
        echo "################################################################"
    } | tee -a "$LOG"

    env \
        -u YERBAS_CUDA_RETUNE \
        -u YERBAS_CUDA_OVERLAP \
        -u YERBAS_GPU_AUTOTUNE \
        -u YERBAS_GPU_VARIANT_AUTOTUNE \
        -u YERBAS_CN_PHASE_RETUNE \
        -u YERBAS_CN_PHASE_THREADS \
        -u YERBAS_GPU_STALE_BATCH_CAP \
        -u YERBAS_GPU_STALE_BATCH_CAP_0 \
        -u YERBAS_GPU_STALE_BATCH_CAP_1 \
        -u YERBAS_GPU_STALE_CROSSOVER_CAP \
        -u YERBAS_CN_DUALHASH_EXPERIMENT \
        -u YERBAS_CN_SPARSE_WARP_HASHES \
        -u YERBAS_CN_BANK8_EXPERIMENT \
        -u YERBAS_CN_MUL4_EXPERIMENT \
        -u YERBAS_CN_PAIRLOAD_EXPERIMENT \
        -u YERBAS_CN_2LANE_TTABLE_EXPERIMENT \
        -u YERBAS_CN_READONLY_TTABLE_EXPERIMENT \
        YERBAS_CUDA_BENCH_RAW_BATCH=1 \
        YERBAS_BENCH_FORCE_CN_TRIPLE="$triple" \
        YERBAS_BENCH_WARMUP_SCANS=1 \
        YERBAS_CN_SETUP_THREADS=32 \
        YERBAS_CN_FINAL_THREADS=128 \
        "$BIN" "$gpu" "${sizes[@]}" 2>&1 | tee -a "$LOG"
}

for gpu in "${GPUS[@]}"; do
    for triple in "${TRIPLES[@]}"; do
        run_case "$gpu" "$triple" 1 "${CANDIDATES[@]}"

        DESC=()
        for ((i=${#CANDIDATES[@]}-1; i>=0; --i)); do
            DESC+=("${CANDIDATES[i]}")
        done
        run_case "$gpu" "$triple" 2 "${DESC[@]}"
    done
done

echo
echo "============================================================"
echo " FRONTIER COMPLETE"
echo "============================================================"
python3 scripts/analyze-large-batch-frontier.py "$LOG" | tee "$LOG.frontier"
echo
echo "Outputs:"
echo "  $LOG"
echo "  $LOG.frontier"
