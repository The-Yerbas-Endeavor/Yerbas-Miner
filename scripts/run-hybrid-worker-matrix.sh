#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${YERBAS_HYBRID_MATRIX_BIN:-$ROOT/build-production-latency/yerbas-miner}"
DURATION="${YERBAS_HYBRID_MATRIX_SECONDS:-900}"
WORKERS="${YERBAS_HYBRID_MATRIX_WORKERS:-4 5 6}"
AFFINITY="${YERBAS_HYBRID_MATRIX_AFFINITY:-unpinned}"
LOG_DIR="$ROOT/logs/hybrid-worker-matrix-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: missing miner binary: $BIN"
    echo "Build it first with: bash scripts/run-production-latency-telemetry.sh"
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
    echo "ERROR: no compatible cached CPU policy found."
    exit 1
fi
IFS=$'\t' read -r CACHE_ROOT CACHE_FILE CACHE_HPS <<< "$CACHE_SELECTION"
export XDG_CACHE_HOME="$CACHE_ROOT"

echo "============================================================"
echo " YERBAS HYBRID CPU WORKER MATRIX"
echo "============================================================"
echo " Binary:       $BIN"
echo " Duration/run: $DURATION seconds"
echo " Workers:      $WORKERS"
echo " Affinity:     $AFFINITY"
echo " CPU cache:    $CACHE_FILE ($CACHE_HPS H/s)"
echo " Logs:         $LOG_DIR"
echo
echo "This is a production contention probe, not a merge benchmark."
echo "It varies only CPU worker count while retaining qualified CN widths."
echo "Press Ctrl+C to stop the matrix."
echo "============================================================"

for workers in $WORKERS; do
    export YERBAS_CPU_WORKERS_OVERRIDE="$workers"
    export YERBAS_CPU_AFFINITY_OVERRIDE="$AFFINITY"
    LOG="$LOG_DIR/workers-${workers}-${AFFINITY}.log"

    echo
    echo "---- workers=$workers affinity=$AFFINITY ----"
    echo "log=$LOG"
    set +e
    script -q -f -e -c "timeout --signal=INT --kill-after=20s ${DURATION}s \"$BIN\"" "$LOG"
    rc=$?
    set -e
    if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then
        echo "WARNING: run exited with status $rc"
    fi
    python3 scripts/analyze-production-latency.py "$LOG" || true
done

unset YERBAS_CPU_WORKERS_OVERRIDE
unset YERBAS_CPU_AFFINITY_OVERRIDE

echo
echo "Matrix complete: $LOG_DIR"

# If this matrix used a stronger qualified default-mode CPU policy from one of
# the benchmark cache roots, make that policy available to ordinary production
# startup too. The helper only promotes when the candidate's recorded H/s is
# higher than the normal cache and preserves the previous production cache.
bash scripts/promote-qualified-cpu-cache.sh || true
