#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-stale-aware}"
BIN="$BUILD_DIR/yerbas-miner"
LOG_DIR="$ROOT/logs"
POLICY_FILE="$ROOT/docs/development/cpu-combo-policy-20260916.txt"
DURATION="${YERBAS_STALE_AB_SECONDS:-900}"
CAP="${YERBAS_STALE_AB_CAP:-5376}"
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

# Reuse the same strongest known CPU/GPU cache root selected by the best-stack
# production runner so this A/B measures the stale cap rather than retuning.
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

# Keep the A/B focused on one variable.
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
unset YERBAS_GPU_STALE_BATCH_CAP || true
unset YERBAS_GPU_STALE_BATCH_CAP_0 || true
unset YERBAS_GPU_STALE_BATCH_CAP_1 || true

run_phase() {
    local label="$1"
    local cap0="$2"
    local cap1="$3"
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"
    local log="$LOG_DIR/stale-batch-ab-${label}-${stamp}.log"

    unset YERBAS_GPU_STALE_BATCH_CAP_0 || true
    unset YERBAS_GPU_STALE_BATCH_CAP_1 || true
    [[ "$cap0" != "0" ]] && export YERBAS_GPU_STALE_BATCH_CAP_0="$cap0"
    [[ "$cap1" != "0" ]] && export YERBAS_GPU_STALE_BATCH_CAP_1="$cap1"

    echo
    echo "============================================================"
    echo " STALE-BATCH A/B PHASE $label"
    echo "============================================================"
    echo " Duration:        ${DURATION}s"
    echo " GPU0 cap:        ${cap0:-0} (0=adaptive)"
    echo " GPU1 cap:        ${cap1:-0} (0=adaptive)"
    echo " CPU combo:       ON"
    echo " CPU cache:       $CACHE_FILE ($CACHE_HPS H/s)"
    echo " CUDA experiments: OFF"
    echo " Log:             $log"
    echo "============================================================"

    set +e
    timeout --signal=INT --kill-after=30s "${DURATION}s" "$BIN" 2>&1 | tee "$log"
    local miner_rc=${PIPESTATUS[0]}
    set -e

    # timeout returns 124 when the requested observation window ends normally.
    if [[ "$miner_rc" -ne 0 && "$miner_rc" -ne 124 && "$miner_rc" -ne 130 ]]; then
        echo "WARNING: miner exited with status $miner_rc"
    fi

    echo
    python3 scripts/analyze-production-latency.py "$log" | tee "$log.analysis"
}

# Crossover design: same pool/job stream, opposite GPU receives the cap.
run_phase "A-gpu1-capped" 0 "$CAP"
sleep 5
run_phase "B-gpu0-capped" "$CAP" 0

echo
echo "============================================================"
echo " CROSSOVER COMPLETE"
echo "============================================================"
echo "Compare the two *.analysis files."
echo "Primary metric: useful H/s per GPU, then stale-hash %, then raw H/s."
echo "A real win should follow the capped role when the cap swaps GPUs."
echo
echo "Logs:"
ls -1t "$LOG_DIR"/stale-batch-ab-*.log "$LOG_DIR"/stale-batch-ab-*.log.analysis 2>/dev/null | head -8
