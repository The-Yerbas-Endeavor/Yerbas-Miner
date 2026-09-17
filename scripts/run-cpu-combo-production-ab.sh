#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/build-cpu-combo-production-ab"
LOG_DIR="$ROOT/logs"
POLICY_FILE="$ROOT/docs/development/cpu-combo-policy-20260916.txt"
mkdir -p "$LOG_DIR"

unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true

python3 scripts/enable-cpu-combo-production-ab.py

git diff --check

echo
printf '%s\n' \
  "============================================================" \
  " YERBAS CPU COMBO FULL-MINER A/B CANDIDATE" \
  "============================================================" \
  " Policy: $POLICY_FILE" \
  " Build:  $BUILD_DIR" \
  ""

# Reuse the strongest known default CPU cache so we do not fall back to the
# stale 393 H/s physical-first policy.
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

echo "CPU cache root : $CACHE_ROOT"
echo "CPU cache file : $CACHE_FILE"
echo "Cached CPU H/s : $CACHE_HPS"

echo
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi

cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

BIN="$BUILD_DIR/yerbas-miner"
LOG="$LOG_DIR/cpu-combo-production-ab-$(date +%Y%m%d-%H%M%S).log"

export YERBAS_CPU_COMBO_POLICY_FILE="$POLICY_FILE"

echo
echo "Starting candidate in foreground."
echo "Expected CPU baseline: 6 workers / batch 16 / 1/4/1/2/4/1 / unpinned / ~432.70 H/s"
echo "Expected combo cache : 6 rules, experimental=yes"
echo "Log                 : $LOG"
echo

script -q -f -e -c "$BIN" "$LOG"
