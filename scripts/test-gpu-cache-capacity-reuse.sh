#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-production-latency}"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/yerbas-miner"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/gpu-cache-capacity-reuse-$STAMP}"
BASELINE_LOG="$LOG_DIR/baseline.log"
REUSE_LOG="$LOG_DIR/reuse.log"

mkdir -p "$LOG_DIR"

if [[ ! -x "$BUILD_DIR/yerbas-miner" ]]; then
    echo "Missing miner binary: $BUILD_DIR/yerbas-miner"
    echo "Build it first with:"
    echo "  cmake -S . -B build-production-latency -DCMAKE_BUILD_TYPE=Release"
    echo '  cmake --build build-production-latency -j"$(nproc)"'
    exit 2
fi

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "Missing GPU cache directory: $CACHE_DIR"
    exit 2
fi

cd "$ROOT_DIR"

unset YERBAS_GPU_AUTOTUNE
unset YERBAS_GPU_VARIANT_AUTOTUNE
unset YERBAS_CUDA_RETUNE
unset YERBAS_CPU_WORKERS_OVERRIDE
unset YERBAS_CPU_AFFINITY_OVERRIDE
unset YERBAS_PRODUCTION_LATENCY_TELEMETRY

echo "==> Baseline startup"
set +e
stdbuf -oL -eL timeout --signal=INT --kill-after=5s 12s \
    "$BUILD_DIR/yerbas-miner" 2>&1 | tee "$BASELINE_LOG"
baseline_rc=${PIPESTATUS[0]}
set -e

current_cap="$(
    sed -n 's/.*\[GPU 0\] adaptive batch capacity.* | max=\([0-9][0-9]*\).*/\1/p' \
        "$BASELINE_LOG" | head -n1
)"

if [[ -z "$current_cap" ]]; then
    echo "Could not determine GPU 0 current capacity from baseline startup."
    exit 3
fi

echo
echo "GPU 0 current capacity: $current_cap"

calibration_exact="$(
    find "$CACHE_DIR" -maxdepth 1 -type f \
        -name "gpu-calibration-rev*-dev0-*-cap${current_cap}-*.txt" \
        -printf '%p\n' | sort -V | tail -n1
)"
variant_exact="$(
    find "$CACHE_DIR" -maxdepth 1 -type f \
        -name "gpu-variant-batch-rev*-dev0-*-cap${current_cap}-*.txt" \
        -printf '%p\n' | sort -V | tail -n1
)"

if [[ -z "$calibration_exact" || -z "$variant_exact" ]]; then
    echo "Exact GPU 0 calibration/variant cache pair not found for cap=$current_cap."
    echo "Calibration: ${calibration_exact:-missing}"
    echo "Variant:     ${variant_exact:-missing}"
    exit 4
fi

sim_cap=$(( current_cap - 128 ))
while (( sim_cap > 0 )); do
    calibration_sim="${calibration_exact/-cap${current_cap}-/-cap${sim_cap}-}"
    variant_sim="${variant_exact/-cap${current_cap}-/-cap${sim_cap}-}"
    if [[ ! -e "$calibration_sim" && ! -e "$variant_sim" ]]; then
        break
    fi
    sim_cap=$(( sim_cap - 128 ))
done

if (( sim_cap <= 0 )); then
    echo "Could not find an unused synthetic capacity filename."
    exit 5
fi

calibration_hold="$calibration_exact.capacity-test-held"
variant_hold="$variant_exact.capacity-test-held"

cleanup()
{
    rm -f -- "${calibration_sim:-}" "${variant_sim:-}"
    if [[ -e "$calibration_hold" ]]; then
        mv -f -- "$calibration_hold" "$calibration_exact"
    fi
    if [[ -e "$variant_hold" ]]; then
        mv -f -- "$variant_hold" "$variant_exact"
    fi
}
trap cleanup EXIT INT TERM

echo
echo "==> Preparing synthetic sibling cache"
echo "    real cap:      $current_cap"
echo "    synthetic cap: $sim_cap"

cp -- "$calibration_exact" "$calibration_sim"

# Variant cache format starts with:
# YERBAS_GPU_VARIANT_BATCH <revision> <cached_capacity> <cached_budget> ...
awk -v cap="$sim_cap" '{ $3 = cap; print }' "$variant_exact" > "$variant_sim.tmp"
mv -- "$variant_sim.tmp" "$variant_sim"

mv -- "$calibration_exact" "$calibration_hold"
mv -- "$variant_exact" "$variant_hold"

echo
echo "==> Forced sibling-cache startup"
set +e
stdbuf -oL -eL timeout --signal=INT --kill-after=5s 18s \
    "$BUILD_DIR/yerbas-miner" 2>&1 | tee "$REUSE_LOG"
reuse_rc=${PIPESTATUS[0]}
set -e

echo
echo "==> Capacity-reuse evidence"
grep -E \
    'GPU 0.*GPU calibration cache loaded|GPU 0.*GPU CN-variant batch cache loaded|GPU 0.*capacity-reuse|GPU 0.*production GPU fast-start|GPU 0.*adaptive batch capacity' \
    "$REUSE_LOG" || true

if grep -q "GPU 0.*GPU calibration cache loaded.*capacity-reuse=${sim_cap}->${current_cap}" "$REUSE_LOG" &&
   grep -q "GPU 0.*GPU CN-variant batch cache loaded.*capacity-reuse=${sim_cap}->${current_cap}" "$REUSE_LOG"; then
    echo
    echo "PASS: GPU 0 safely reused both calibration and variant caches across capacity change."
    result=0
else
    echo
    echo "FAIL: expected both GPU 0 capacity-reuse markers."
    result=6
fi

echo
echo "Baseline exit code: $baseline_rc"
echo "Reuse exit code:    $reuse_rc"
echo "Logs:"
echo "  $BASELINE_LOG"
echo "  $REUSE_LOG"

exit "$result"
