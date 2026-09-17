#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/build-cpu-combo-production-ab/yerbas-miner"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: candidate binary is missing: $BIN"
    echo "Run scripts/run-cpu-combo-production-ab.sh once to build it."
    exit 1
fi

# Exact same binary as the combo-policy run, but policy disabled.
unset YERBAS_CPU_COMBO_POLICY_FILE || true
unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true

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

LOG="$LOG_DIR/cpu-combo-production-baseline-$(date +%Y%m%d-%H%M%S).log"

echo "============================================================"
echo " YERBAS CPU COMBO FULL-MINER BASELINE"
echo "============================================================"
echo " Binary:         $BIN"
echo " Combo policy:   OFF"
echo " CPU cache root: $CACHE_ROOT"
echo " CPU cache file: $CACHE_FILE"
echo " Cached CPU H/s: $CACHE_HPS"
echo " Log:            $LOG"
echo ""
echo "Starting baseline in foreground. Press Ctrl+C to stop."
echo "============================================================"
echo

script -q -f -e -c "$BIN" "$LOG"
