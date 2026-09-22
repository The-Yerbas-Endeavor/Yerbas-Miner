#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-midsize-phase}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

BATCH=7168
TRIPLES=(
  "dark,lite,turtlelite"
  "darklite,lite,turtle"
)
REPEATS="${YERBAS_7168_AB_REPEATS:-4}"

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
LOG="$LOG_DIR/batch7168-setup32-vs128-${STAMP}.log"

{
  echo "============================================================"
  echo " YERBAS BATCH-7168 SETUP GEOMETRY A/B"
  echo "============================================================"
  echo " GPUs:        ${GPUS[*]}"
  echo " Batch:       $BATCH"
  echo " Triples:     ${TRIPLES[*]}"
  echo " Repeats:     $REPEATS"
  echo " A/B:         setup 32 vs 128; final fixed at 128"
  echo " Order:       alternates each repeat"
  echo " Scratchpad:  1 MiB/hash"
  echo " Cache writes: DISABLED"
  echo " Production policy: unchanged"
  echo " Log:         $LOG"
  echo "============================================================"
} | tee "$LOG"

run_case() {
  local gpu="$1"
  local triple="$2"
  local repeat="$3"
  local setup_threads="$4"

  {
    echo
    echo "################################################################"
    echo "### GPU $gpu | triple $triple | repeat=$repeat | setup=$setup_threads"
    echo "################################################################"
  } | tee -a "$LOG"

  env \
    -u YERBAS_CUDA_RETUNE \
    -u YERBAS_GPU_AUTOTUNE \
    -u YERBAS_GPU_VARIANT_AUTOTUNE \
    -u YERBAS_GPU_STALE_BATCH_CAP \
    -u YERBAS_GPU_STALE_BATCH_CAP_0 \
    -u YERBAS_GPU_STALE_BATCH_CAP_1 \
    -u YERBAS_GPU_STALE_BATCH_MIN_TUNED \
    -u YERBAS_GPU_STALE_CROSSOVER_CAP \
    -u YERBAS_GPU_STALE_CROSSOVER_MIN_TUNED \
    -u YERBAS_CN_PHASE_RETUNE \
    -u YERBAS_CN_PHASE_THREADS \
    YERBAS_CN_PHASE_NOSAVE=1 \
    YERBAS_CN_SETUP_THREADS="$setup_threads" \
    YERBAS_CN_FINAL_THREADS=128 \
    YERBAS_CUDA_BENCH_RAW_BATCH=1 \
    YERBAS_CUDA_BENCH_SCRATCHPAD_STRIDE=1048576 \
    YERBAS_BENCH_FORCE_CN_TRIPLE="$triple" \
    YERBAS_BENCH_WARMUP_SCANS=1 \
    "$BIN" "$gpu" "$BATCH" 2>&1 | tee -a "$LOG"
}

for gpu in "${GPUS[@]}"; do
  for triple in "${TRIPLES[@]}"; do
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
      if (( repeat % 2 == 1 )); then
        run_case "$gpu" "$triple" "$repeat" 32
        run_case "$gpu" "$triple" "$repeat" 128
      else
        run_case "$gpu" "$triple" "$repeat" 128
        run_case "$gpu" "$triple" "$repeat" 32
      fi
    done
  done
done

echo
echo "============================================================"
echo " BATCH-7168 SETUP A/B COMPLETE"
echo "============================================================"
python3 scripts/analyze-7168-setup-ab.py "$LOG" | tee "$LOG.summary"

echo
echo "Outputs:"
echo "  $LOG"
echo "  $LOG.summary"
