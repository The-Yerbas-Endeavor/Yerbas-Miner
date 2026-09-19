#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${YERBAS_CN_PHASE_BIN:-$ROOT/build-production-latency/yerbas-miner}"
DURATION="${YERBAS_CN_PHASE_SECONDS:-120}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${YERBAS_CN_PHASE_LOG_DIR:-$ROOT/logs/cn-phase-one-shot-$STAMP}"
RUN_LOG="$LOG_DIR/miner.log"
SUMMARY="$LOG_DIR/phase-summary.txt"

mkdir -p "$LOG_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: missing miner binary: $BIN"
    exit 1
fi

# Preserve normal cached production policy. We only enable the built-in one-shot
# phase timing path; no retune is requested and no CUDA cache is invalidated.
unset YERBAS_CPU_RETUNE
unset YERBAS_CUDA_RETUNE
unset YERBAS_GPU_AUTOTUNE
unset YERBAS_GPU_VARIANT_AUTOTUNE
unset YERBAS_CPU_WORKERS_OVERRIDE
unset YERBAS_CPU_AFFINITY_OVERRIDE
unset YERBAS_PRODUCTION_LATENCY_TELEMETRY
unset YERBAS_CUDA_OVERLAP
export YERBAS_DIAGNOSTICS=1

echo "============================================================"
echo " YERBAS CN PRODUCTION PHASE ONE-SHOT"
echo "============================================================"
echo " Binary:   $BIN"
echo " Duration: $DURATION seconds"
echo " Log:      $RUN_LOG"
echo
echo "No retune and no rebuild. The first production invocation of each"
echo "GPU/CN-variant pair records setup/loop/final timing once."
echo "============================================================"

set +e
stdbuf -oL -eL timeout --signal=INT --kill-after=10s "${DURATION}s"     "$BIN" 2>&1 | tee "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e

grep -E '^\[CUDA CN phase one-shot\]' "$RUN_LOG" > "$SUMMARY" || true

echo
echo "============================================================"
echo " CN PHASE SUMMARY"
echo "============================================================"
if [[ -s "$SUMMARY" ]]; then
    cat "$SUMMARY"
else
    echo "No phase one-shot samples were captured."
fi

echo
echo "Miner exit code: $rc"
case "$rc" in
    0|124|130) ;;
    *) echo "WARNING: unexpected miner exit status $rc" ;;
esac
echo "Summary: $SUMMARY"
echo "Full log: $RUN_LOG"

# A timeout is expected for this bounded diagnostic.
if [[ "$rc" -eq 124 || "$rc" -eq 130 ]]; then
    exit 0
fi
exit "$rc"
