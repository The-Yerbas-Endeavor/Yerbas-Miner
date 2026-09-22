#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-stale-aware}"
BIN="$BUILD_DIR/yerbas-miner"
LOG_DIR="$ROOT/logs"
POLICY_FILE="$ROOT/docs/development/cpu-combo-policy-20260916.txt"
DURATION="${YERBAS_STALE_LIVE_SECONDS:-2700}"
CAP="${YERBAS_STALE_LIVE_CAP:-5376}"
MIN_TUNED="${YERBAS_STALE_LIVE_MIN_TUNED:-0}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building stale-aware production miner..."
cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: miner binary not found: $BIN"
    exit 1
fi

if [[ ! -f "$POLICY_FILE" ]]; then
    echo "ERROR: CPU combo policy not found: $POLICY_FILE"
    exit 1
fi

NORMAL_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
CACHE_SELECTION="$({
python3 - "$ROOT" "$NORMAL_CACHE_ROOT" <<'PY'
import glob
import os
import sys

root, normal = sys.argv[1:]
roots = [
    os.path.join(root, ".bench-cache", "4lane-splitmul"),
    os.path.join(root, ".bench-cache", "workday-4lane-splitmul"),
    normal,
]
best = None
seen = set()
for cache_root in roots:
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    if cache_root in seen:
        continue
    seen.add(cache_root)
    for path in glob.glob(os.path.join(cache_root, "yerbas-miner", "cpu-policy-rev*-default.txt")):
        try:
            parts = open(path, encoding="utf-8").read().split()
            if len(parts) < 13 or parts[0] != "YERBAS_CPU_POLICY":
                continue
            throughput = float(parts[-1])
        except Exception:
            continue
        row = (throughput, cache_root, path)
        if best is None or row[0] > best[0]:
            best = row
if best is None:
    raise SystemExit(1)
print(f"{best[1]}\t{best[2]}\t{best[0]:.6f}")
PY
} 2>/dev/null)"

if [[ -z "$CACHE_SELECTION" ]]; then
    echo "ERROR: no compatible CPU cache found."
    exit 1
fi

IFS=$'\t' read -r CACHE_ROOT CACHE_FILE CACHE_HPS <<< "$CACHE_SELECTION"
export XDG_CACHE_HOME="$CACHE_ROOT"
export YERBAS_CPU_COMBO_POLICY_FILE="$POLICY_FILE"
export YERBAS_PRODUCTION_LATENCY_TELEMETRY=1
export YERBAS_GPU_STALE_CROSSOVER_CAP="$CAP"
export YERBAS_GPU_STALE_CROSSOVER_MIN_TUNED="$MIN_TUNED"

# Keep this as a one-variable production experiment.
unset YERBAS_GPU_STALE_BATCH_CAP || true
unset YERBAS_GPU_STALE_BATCH_CAP_0 || true
unset YERBAS_GPU_STALE_BATCH_CAP_1 || true
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

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/stale-live-crossover-${STAMP}.log"

echo
echo "============================================================"
echo " YERBAS LIVE STALE-BATCH CROSSOVER"
echo "============================================================"
echo " Duration:         ${DURATION}s"
echo " Cap:              $CAP"
echo " Minimum tuned:    $MIN_TUNED (0 = any batch above cap)"
echo " Role switching:   every Stratum generation"
echo " CPU combo policy: ON"
echo " CPU cache:        $CACHE_FILE ($CACHE_HPS H/s)"
echo " CUDA experiments: OFF"
echo " Log:              $LOG"
echo
if [[ "$MIN_TUNED" -gt 0 ]]; then
    echo "Only generations where the adaptive GPU selects >=$MIN_TUNED hashes"
else
    echo "Only generations where the adaptive GPU selects >$CAP hashes"
fi
echo "count toward the final capped-vs-adaptive comparison."
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
python3 scripts/analyze-stale-crossover.py "$LOG" | tee "$LOG.crossover"
echo
python3 scripts/analyze-production-latency.py "$LOG" | tee "$LOG.analysis"

echo
echo "Outputs:"
echo "  $LOG"
echo "  $LOG.crossover"
echo "  $LOG.analysis"
