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
    print(f"{max(1, logical-1)}\t0:0,1:0\tunknown")
    raise SystemExit

ordered = sorted(cores.items())
reserve_key, reserve_cpus = ordered[-1]
physical = len(ordered)
workers = max(1, physical - 1)
if len(reserve_cpus) == 1:
    reserve_cpus = reserve_cpus * 2

mapping = f"0:{reserve_cpus[0]},1:{reserve_cpus[1]}"
label = f"package{reserve_key[0]}-core{reserve_key[1]}-cpus{','.join(map(str,reserve_cpus))}"
print(f"{workers}\t{mapping}\t{label}")
PY
} 2>/dev/null)"

IFS=$'\t' read -r RESERVED_WORKERS GPU_MAP RESERVED_LABEL <<< "$TOPOLOGY"

echo "============================================================"
echo " YERBAS HYBRID HOST-AFFINITY A/B"
echo "============================================================"
echo " Binary:          $BIN"
echo " Duration/run:    $DURATION seconds"
echo " Reserved core:   $RESERVED_LABEL"
echo " Reserved workers:$RESERVED_WORKERS"
echo " GPU host map:    $GPU_MAP"
echo " Logs:            $LOG_DIR"
echo
echo "A = normal qualified policy (6-worker cache result / unpinned host)"
echo "B = one physical core reserved from CPU mining; GPU host threads pinned there"
echo "============================================================"

run_one() {
    local label="$1"
    local workers="$2"
    local affinity_policy="$3"
    local gpu_map="$4"
    local log="$LOG_DIR/$label.log"

    export YERBAS_PRODUCTION_LATENCY_TELEMETRY=1
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

unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true
unset YERBAS_CPU_COMBO_POLICY_FILE || true

# Reuse whichever qualified default CPU cache the latency runner selected.
# The miner's normal cache loading still supplies the CN widths/lane policy.
run_one "A-normal-unpinned" 6 unpinned ""
run_one "B-reserved-core" "$RESERVED_WORKERS" physical-first "$GPU_MAP"

unset YERBAS_CPU_WORKERS_OVERRIDE
unset YERBAS_CPU_AFFINITY_OVERRIDE
unset YERBAS_GPU_HOST_CPU_MAP

echo
echo "Affinity A/B complete: $LOG_DIR"
