#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/build-production-latency"
BIN="$BUILD_DIR/yerbas-miner"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Configuring CUDA telemetry build..."
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

echo "Building yerbas-miner (incremental after the first build)..."
cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: miner binary was not produced: $BIN"
    exit 1
fi

unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true
unset YERBAS_CPU_COMBO_POLICY_FILE || true
export YERBAS_PRODUCTION_LATENCY_TELEMETRY=1

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

LOG="$LOG_DIR/production-latency-$(date +%Y%m%d-%H%M%S).log"

echo "============================================================"
echo " YERBAS PRODUCTION LATENCY TELEMETRY"
echo "============================================================"
echo " Branch target:  feature/production-latency-telemetry"
echo " Binary:         $BIN"
echo " Scheduling:     unchanged"
echo " Combo policy:   OFF (clean latency baseline)"
echo " CPU cache root: $CACHE_ROOT"
echo " CPU cache file: $CACHE_FILE"
echo " Cached CPU H/s: $CACHE_HPS"
echo " Telemetry:      ON"
echo " Log:            $LOG"
echo
echo "Starting in FOREGROUND. Press Ctrl+C to stop."
echo "============================================================"
echo

script -q -f -e -c "$BIN" "$LOG"

echo
echo "Latency summary:"
python3 scripts/analyze-production-latency.py "$LOG"
