#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-stale-aware-production}"
BIN="$BUILD_DIR/yerbas-miner"
LOG_DIR="$ROOT/logs"
POLICY_FILE="$ROOT/docs/development/cpu-combo-policy-20260916.txt"
DURATION="${YERBAS_STALE_PRODUCTION_SECONDS:-3600}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$POLICY_FILE" ]]; then
    echo "ERROR: CPU combo policy not found: $POLICY_FILE"
    exit 1
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building stale-aware production candidate..."
cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: miner binary not found: $BIN"
    exit 1
fi

# Validate the shipped/default policy, not an experiment inherited from shell.
unset YERBAS_GPU_STALE_BATCH_CAP || true
unset YERBAS_GPU_STALE_BATCH_CAP_0 || true
unset YERBAS_GPU_STALE_BATCH_CAP_1 || true
unset YERBAS_GPU_STALE_BATCH_MIN_TUNED || true
unset YERBAS_GPU_STALE_CROSSOVER_CAP || true
unset YERBAS_GPU_STALE_CROSSOVER_MIN_TUNED || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true
unset YERBAS_CN_PHASE_RETUNE || true
unset YERBAS_CN_PHASE_THREADS || true
unset YERBAS_CN_SETUP_THREADS || true
unset YERBAS_CN_FINAL_THREADS || true
unset YERBAS_CN_DUALHASH_EXPERIMENT || true
unset YERBAS_CN_DUALHASH_ONLY || true
unset YERBAS_CN_DUALHASH_THREADS || true
unset YERBAS_CN_SPARSE_WARP_HASHES || true
unset YERBAS_CN_SPARSE_WARP_ONLY || true
unset YERBAS_CN_BANK8_EXPERIMENT || true
unset YERBAS_CN_MUL4_EXPERIMENT || true
unset YERBAS_CN_PAIRLOAD_EXPERIMENT || true
unset YERBAS_CN_2LANE_TTABLE_EXPERIMENT || true
unset YERBAS_CN_READONLY_TTABLE_EXPERIMENT || true
unset YERBAS_CUDA_BENCH_RAW_BATCH || true
unset YERBAS_CUDA_BENCH_STAGE_LOCAL || true
unset YERBAS_CUDA_BENCH_CN_CHUNK || true
unset YERBAS_CUDA_BENCH_SCRATCHPAD_STRIDE || true

export YERBAS_CPU_COMBO_POLICY_FILE="$POLICY_FILE"
export YERBAS_PRODUCTION_LATENCY_TELEMETRY=1

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/stale-aware-production-${STAMP}.log"

echo
echo "============================================================"
echo " YERBAS STALE-AWARE PRODUCTION CANDIDATE"
echo "============================================================"
echo " Duration:         ${DURATION}s"
echo " CPU combo policy: ON"
echo " GPU experiments:  OFF"
echo " Stale policy:     shipped/default hardware policy"
echo " Telemetry:        ON"
echo " Log:              $LOG"
echo
echo "Expected on GTX 1080 Ti:"
echo "  stale-aware production policy | cap=3584 | minimum-tuned=6272"
echo "  large tuned rotations >=6272 -> 3584"
echo "  normal 3584/5376 rotations unchanged"
echo "============================================================"
echo

set +e
timeout --signal=INT --kill-after=30s "${DURATION}s" "$BIN" 2>&1 | tee "$LOG"
MINER_RC=${PIPESTATUS[0]}
set -e

if [[ "$MINER_RC" -ne 0 && "$MINER_RC" -ne 124 && "$MINER_RC" -ne 130 ]]; then
    echo "WARNING: miner exited with status $MINER_RC"
fi

echo
python3 scripts/analyze-production-latency.py "$LOG" | tee "$LOG.analysis"
echo
python3 scripts/analyze-rotation-perf.py "$LOG" | tee "$LOG.rotation"

echo
echo "Outputs:"
echo "  $LOG"
echo "  $LOG.analysis"
echo "  $LOG.rotation"
