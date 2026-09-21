#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/build-production-latency"
BIN="$BUILD_DIR/yerbas-miner"
LOG_DIR="$ROOT/logs"
POLICY_FILE="$ROOT/docs/development/cpu-combo-policy-20260916.txt"
mkdir -p "$LOG_DIR"

if [[ ! -f "$POLICY_FILE" ]]; then
    echo "ERROR: CPU combo policy not found: $POLICY_FILE"
    exit 1
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Configuring CUDA production build..."
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building yerbas-miner..."
cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: miner binary was not produced: $BIN"
    exit 1
fi

# Production validation must not inherit any benchmark-only CUDA experiment.
unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true
unset YERBAS_CN_PHASE_RETUNE || true
unset YERBAS_CN_PHASE_THREADS || true
unset YERBAS_CN_SETUP_THREADS || true
unset YERBAS_CN_FINAL_THREADS || true
unset YERBAS_BENCH_FORCE_CN_VARIANT || true
unset YERBAS_BENCH_FORCE_CN_TRIPLE || true
unset YERBAS_CUDA_BENCH_RAW_BATCH || true
unset YERBAS_CUDA_BENCH_STAGE_LOCAL || true
unset YERBAS_CUDA_BENCH_CN_CHUNK || true
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

export YERBAS_CPU_COMBO_POLICY_FILE="$POLICY_FILE"
export YERBAS_PRODUCTION_LATENCY_TELEMETRY=1

# Reuse the strongest known default CPU cache so this run measures the proven
# production stack instead of spending startup time relearning CPU policy.
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
    echo "ERROR: no compatible cached default CPU policy found."
    exit 1
fi

IFS=$'\t' read -r CACHE_ROOT CACHE_FILE CACHE_HPS <<< "$CACHE_SELECTION"
export XDG_CACHE_HOME="$CACHE_ROOT"

LOG="$LOG_DIR/best-stack-production-$(date +%Y%m%d-%H%M%S).log"

echo
echo "============================================================"
echo " YERBAS BEST-KNOWN-STACK PRODUCTION VALIDATION"
echo "============================================================"
echo " Binary:           $BIN"
echo " CPU combo policy: ON"
echo " Policy file:      $POLICY_FILE"
echo " CPU cache root:   $CACHE_ROOT"
echo " CPU cache file:   $CACHE_FILE"
echo " Cached CPU H/s:   $CACHE_HPS"
echo " GPU experiments:  OFF"
echo " GPU policy:       cache-first production"
echo " Telemetry:        ON"
echo " Log:              $LOG"
echo
echo "Recommended validation window: 6-12 hours."
echo "Press Ctrl+C when you want to stop."
echo "============================================================"
echo

script -q -f -e -c "$BIN" "$LOG"

echo
echo "Production latency summary:"
python3 scripts/analyze-production-latency.py "$LOG" || true

echo
echo "Rotation summary:"
python3 scripts/analyze-rotation-perf.py "$LOG" || true
