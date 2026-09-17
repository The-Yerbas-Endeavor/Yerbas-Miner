#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/build-cpu-combo"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# This diagnostic must consume the already-proven startup CPU policy. Do not let
# stale developer controls trigger fresh tuning or live rotation learning.
unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S . -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DYERBAS_ENABLE_CUDA=OFF \
        -DYERBAS_NATIVE_CPU=OFF
fi

cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

BIN="$BUILD_DIR/yerbas-miner"
LOG="$LOG_DIR/cpu-cn-combo-$(date +%Y%m%d-%H%M%S).log"

printf '%s\n' \
  "============================================================" \
  " YERBAS CPU CN-COMBINATION BENCHMARK" \
  "============================================================" \
  " Binary: $BIN" \
  " Log:    $LOG" \
  "" \
  " Uses the existing cached CPU policy as the baseline." \
  " Tests all 20 unordered three-CryptoNight combinations." \
  " Does not initialize CUDA, connect to the pool, or write a" \
  " production CPU combination policy." \
  "============================================================" \
  ""

export YERBAS_CPU_COMBO_BENCH=1
export YERBAS_CPU_COMBO_PASSES="${YERBAS_CPU_COMBO_PASSES:-5}"

set +e
script -q -f -e -c "$BIN" "$LOG"
STATUS=$?
set -e

echo
echo "================ BENCHMARK SUMMARY ================"ngrep -E '^\[CPU combo result\]|^Combinations tested|^Promoted candidates|^Equal-combo|^Projected CPU gain|^ERROR:|^Fatal:' "$LOG" || true

echo "Log: $LOG"
exit "$STATUS"
