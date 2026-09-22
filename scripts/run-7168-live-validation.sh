#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-7168-live}"
BIN="$BUILD_DIR/yerbas-miner"
LOG_DIR="$ROOT/logs"
POLICY_FILE="$ROOT/docs/development/cpu-combo-policy-20260916.txt"
DURATION="${YERBAS_7168_LIVE_SECONDS:-7200}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$POLICY_FILE" ]]; then
  echo "ERROR: CPU combo policy not found: $POLICY_FILE" >&2
  exit 1
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building 7168 live-validation candidate..."
cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: miner binary not found: $BIN" >&2
  exit 1
fi

# Validate the branch's default policy. Clear all geometry/tuning overrides.
unset YERBAS_CN_PHASE_RETUNE || true
unset YERBAS_CN_PHASE_THREADS || true
unset YERBAS_CN_SETUP_THREADS || true
unset YERBAS_CN_FINAL_THREADS || true
unset YERBAS_CN_PHASE_NOSAVE || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true
unset YERBAS_GPU_STALE_BATCH_CAP || true
unset YERBAS_GPU_STALE_BATCH_CAP_0 || true
unset YERBAS_GPU_STALE_BATCH_CAP_1 || true
unset YERBAS_GPU_STALE_BATCH_MIN_TUNED || true
unset YERBAS_GPU_STALE_CROSSOVER_CAP || true
unset YERBAS_GPU_STALE_CROSSOVER_MIN_TUNED || true

export YERBAS_CPU_COMBO_POLICY_FILE="$POLICY_FILE"
export YERBAS_PRODUCTION_LATENCY_TELEMETRY=1

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/batch7168-live-${STAMP}.log"

echo
echo "============================================================"
echo " YERBAS BATCH-7168 LIVE VALIDATION"
echo "============================================================"
echo " Duration:         ${DURATION}s (~$((DURATION / 3600))h)"
echo " Expected 7168:    setup=32 final=128 on GTX 1080 Ti"
echo " 5376 behavior:    unchanged"
echo " Stale-aware cap:  shipped/main policy"
echo " Telemetry:        ON"
echo " Log:              $LOG"
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
echo "7168 geometry observations:"
grep -E 'phase geometry safe default .*batch=7168|phase geometry cache loaded .*batch=7168' "$LOG" | head -80 || true

echo
echo "Outputs:"
echo "  $LOG"
echo "  $LOG.analysis"
echo "  $LOG.rotation"
