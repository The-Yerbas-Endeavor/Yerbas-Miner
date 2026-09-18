#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${YERBAS_HYBRID_AFFINITY_BIN:-$ROOT/build-production-latency/yerbas-miner}"
DURATION="${YERBAS_HYBRID_AFFINITY_SECONDS:-900}"
LOG_DIR="$ROOT/logs/hybrid-affinity-ab-$(date +%Y%m%d-%H%M%S)"
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
            workers = int(parts[2])
            throughput = float(parts[-1])
        except Exception:
            continue
        row = (throughput, cache_root, path, workers)
        if best is None or row[0] > best[0]:
            best = row

if best is None:
    raise SystemExit(1)

print(f"{best[1]}\t{best[2]}\t{best[0]:.6f}\t{best[3]}")
PY
} 2>/dev/null)"

if [[ -z "$CACHE_SELECTION" ]]; then
    echo "ERROR: no compatible cached CPU policy found."
    exit 1
fi

IFS=$'\t' read -r CACHE_ROOT CACHE_FILE CACHE_HPS BASELINE_WORKERS <<< "$CACHE_SELECTION"
export XDG_CACHE_HOME="$CACHE_ROOT"

TOPOLOGY="$({
python3 - <<'PY'
from collections import defaultdict
from pathlib import Path
import os

logical = os.cpu_count() or 1
cores = defaultdict(list)

for cpu in range(logical):
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = int((base / "physical_package_id").read_text().strip())
        core = int((base / "core_id").read_text().strip())
    except Exception:
        continue
    cores[(package, core)].append(cpu)

if not cores:
    workers = max(1, logical - 1)
    print(f"{workers}\t0:0,1:0\tunknown")
    raise SystemExit

ordered = sorted(cores.items())
reserve_key, reserve_cpus = ordered[-1]
physical = len(ordered)
workers = max(1, physical - 1)

if len(reserve_cpus) == 1:
    reserve_cpus = reserve_cpus * 2

mapping = f"0:{reserve_cpus[0]},1:{reserve_cpus[1]}"
label = (
    f"package{reserve_key[0]}-core{reserve_key[1]}-"
    f"cpus{','.join(map(str, reserve_cpus))}"
)
print(f"{workers}\t{mapping}\t{label}")
PY
} 2>/dev/null)"

if [[ -z "$TOPOLOGY" ]]; then
    echo "ERROR: unable to determine CPU topology."
    exit 1
fi

IFS=$'\t' read -r RESERVED_WORKERS GPU_MAP RESERVED_LABEL <<< "$TOPOLOGY"

echo "============================================================"
echo " YERBAS HYBRID HOST-AFFINITY A/B"
echo "============================================================"
echo " Binary:           $BIN"
echo " Duration/run:     $DURATION seconds"
echo " Baseline workers: $BASELINE_WORKERS"
echo " Reserved workers: $RESERVED_WORKERS"
echo " Reserved core:    $RESERVED_LABEL"
echo " GPU host map:     $GPU_MAP"
echo " CPU cache:        $CACHE_FILE ($CACHE_HPS H/s)"
echo " Logs:             $LOG_DIR"
echo
echo "A = qualified worker count, CPU unpinned, GPU host threads unpinned"
echo "B = one physical core reserved; GPU host threads pinned to that core"
echo "============================================================"

run_one() {
    local label="$1"
    local workers="$2"
    local affinity_policy="$3"
    local gpu_map="$4"
    local log="$LOG_DIR/$label.log"

    export YERBAS_CPU_WORKERS_OVERRIDE="$workers"
    export YERBAS_CPU_AFFINITY_OVERRIDE="$affinity_policy"

    if [[ -n "$gpu_map" ]]; then
        export YERBAS_GPU_HOST_CPU_MAP="$gpu_map"
    else
        unset YERBAS_GPU_HOST_CPU_MAP || true
    fi

    echo
    echo "---- $label ----"
    echo "workers=$workers affinity=$affinity_policy gpu_host_map=${gpu_map:-unpinned}"
    echo "log=$log"

    set +e
    script -q -f -e -c "timeout --signal=INT --kill-after=20s ${DURATION}s \"$BIN\"" "$log"
    rc=$?
    set -e

    if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then
        echo "WARNING: $label exited with status $rc"
    fi

    python3 scripts/analyze-production-latency.py "$log" || true
}

run_one "A-normal-unpinned" "$BASELINE_WORKERS" unpinned ""
run_one "B-reserved-core" "$RESERVED_WORKERS" physical-first "$GPU_MAP"

unset YERBAS_CPU_WORKERS_OVERRIDE
unset YERBAS_CPU_AFFINITY_OVERRIDE
unset YERBAS_GPU_HOST_CPU_MAP

echo
echo "Affinity A/B complete: $LOG_DIR"
