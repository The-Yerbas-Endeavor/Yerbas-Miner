#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-production-latency}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/gpu-cache-startup-$STAMP}"
BUILD_LOG="$LOG_DIR/build.log"
RUN_LOG="$LOG_DIR/startup.log"
CACHE_LOG="$LOG_DIR/cache-files.txt"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

echo "==> GPU cache startup diagnostic"
echo "    repo:  $ROOT_DIR"
echo "    build: $BUILD_DIR"
echo "    logs:  $LOG_DIR"
echo

echo "==> Configuring"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$BUILD_LOG"
configure_rc=${PIPESTATUS[0]}
if (( configure_rc != 0 )); then
    echo
    echo "CONFIGURE FAILED rc=$configure_rc"
    echo "Log: $BUILD_LOG"
    exit "$configure_rc"
fi

echo
echo "==> Building"
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tee -a "$BUILD_LOG"
build_rc=${PIPESTATUS[0]}
if (( build_rc != 0 )); then
    echo
    echo "BUILD FAILED rc=$build_rc"
    echo "Log: $BUILD_LOG"
    exit "$build_rc"
fi

echo
echo "==> Existing GPU cache files"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/yerbas-miner"
if [[ -d "$CACHE_DIR" ]]; then
    find "$CACHE_DIR" -maxdepth 1 -type f \
        \( -name 'gpu-calibration-*' -o -name 'gpu-variant-batch-*' \) \
        -printf '%f\n' | sort | tee "$CACHE_LOG"
else
    echo "(cache directory does not exist: $CACHE_DIR)" | tee "$CACHE_LOG"
fi

unset YERBAS_GPU_AUTOTUNE
unset YERBAS_GPU_VARIANT_AUTOTUNE
unset YERBAS_CUDA_RETUNE
unset YERBAS_CPU_WORKERS_OVERRIDE
unset YERBAS_CPU_AFFINITY_OVERRIDE
unset YERBAS_PRODUCTION_LATENCY_TELEMETRY

echo
echo "==> Starting miner for 30 seconds"
echo "    stdout/stderr are line-buffered so an early crash is preserved."
echo

set +e
stdbuf -oL -eL timeout --signal=INT --kill-after=10s 30s \
    "$BUILD_DIR/yerbas-miner" 2>&1 | tee "$RUN_LOG"
run_rc=${PIPESTATUS[0]}
set -e

echo
echo "==> Startup result"
echo "exit_code=$run_rc"

case "$run_rc" in
    0)
        echo "Miner exited normally."
        ;;
    124)
        echo "Expected diagnostic timeout reached; startup survived."
        ;;
    130)
        echo "Miner exited on SIGINT."
        ;;
    134)
        echo "Miner aborted (SIGABRT)."
        ;;
    139)
        echo "Miner crashed with SIGSEGV."
        ;;
    *)
        echo "Miner exited with rc=$run_rc."
        ;;
esac

echo
echo "==> Relevant startup lines"
grep -E \
    'GPU calibration cache loaded|GPU CN-variant batch cache loaded|capacity-reuse|production GPU fast-start|adaptive batch capacity|CUDA|Fatal|terminate|error|Error|failed|FAIL' \
    "$RUN_LOG" || true

echo
echo "Logs:"
echo "  $BUILD_LOG"
echo "  $RUN_LOG"
echo "  $CACHE_LOG"

exit "$run_rc"
