#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANDIDATE_REF="origin/feature/cn-4lane-splitmul"
BUILD_DIR="$ROOT/build-4lane-splitmul"
LOG_DIR="$ROOT/logs"

# Prefer the completed profile produced by the previous workday's long first-run
# tuning session. Fall back to the user's normal production cache when needed.
NORMAL_XDG_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
NORMAL_CACHE="$NORMAL_XDG_ROOT/yerbas-miner"
PREVIOUS_TUNED_CACHE="$ROOT/.bench-cache/4lane-splitmul/yerbas-miner"
WORK_XDG_ROOT="$ROOT/.bench-cache/workday-4lane-splitmul"
WORK_CACHE="$WORK_XDG_ROOT/yerbas-miner"

has_complete_profile() {
    local cache="$1"
    [[ -d "$cache" ]] && \
    compgen -G "$cache/cpu-policy-v6*" >/dev/null && \
    compgen -G "$cache/gpu-calibration-v2*" >/dev/null && \
    compgen -G "$cache/gpu-variant-batch-v2*" >/dev/null
}

mkdir -p "$LOG_DIR"
git fetch origin

if has_complete_profile "$PREVIOUS_TUNED_CACHE"; then
    SOURCE_CACHE="$PREVIOUS_TUNED_CACHE"
    SOURCE_LABEL="previous completed workday autotune"
elif has_complete_profile "$NORMAL_CACHE"; then
    SOURCE_CACHE="$NORMAL_CACHE"
    SOURCE_LABEL="normal production tuning cache"
else
    echo "ERROR: no completed tuning profile was found."
    echo "Checked:"
    echo "  $PREVIOUS_TUNED_CACHE"
    echo "  $NORMAL_CACHE"
    echo "Refusing to start a run that could stop at the first-run autotune prompt."
    exit 1
fi

echo
printf '%s\n' "============================================================" \
              " CLEAN WORKDAY CANDIDATE TEST" \
              " - reuses completed CPU/GPU tuning" \
              " - retests only CN-Fast phase-2 selector" \
              " - requires >=8% candidate gain on BOTH GPUs" \
              " - starts the miner attached in foreground" \
              "============================================================"
echo "Tuning source: $SOURCE_LABEL"
echo "Cache source:  $SOURCE_CACHE"
echo

git switch --detach "$CANDIDATE_REF"

# Incremental build; avoid another unnecessary full clean CUDA rebuild.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi
cmake --build "$BUILD_DIR" --target cuda-batch-benchmark yerbas-miner -j"$(nproc)"

# Clone the already-completed tuning profile. The original cache is never
# modified by this experiment.
rm -rf "$WORK_XDG_ROOT"
mkdir -p "$WORK_XDG_ROOT"
cp -a "$SOURCE_CACHE" "$WORK_CACHE"

# Variant index 2 is CN-Fast. Invalidate only selector/geometry decisions whose
# implementation is changed by this branch. Hardware calibration, CPU policy,
# per-variant batch sizing, setup/final backends and every other CN variant stay
# warm. This is the key difference from the previous four-hour prompt incident.
rm -f "$WORK_CACHE"/cn-production-rev2-v2-*.txt
rm -f "$WORK_CACHE"/cn-phase2-rev2-v2-*.txt
rm -f "$WORK_CACHE"/cn-block-rev1-v2-m444-*.txt
rm -f "$WORK_CACHE"/cn-geometry-rev2-v2-m444-*.txt

PRIME_PREFIX="$LOG_DIR/4lane-splitmul-clean-prime"

for GPU in 0 1; do
    env \
      XDG_CACHE_HOME="$WORK_XDG_ROOT" \
      YERBAS_BENCH_FORCE_CN_FAST=1 \
      YERBAS_CUDA_OVERLAP=0 \
      "$BUILD_DIR/cuda-batch-benchmark" "$GPU" 3584 \
      > "${PRIME_PREFIX}-gpu${GPU}.log" 2>&1
done

echo "================ CN-FAST PRIME ================"
grep -E \
'production selector|candidate=baseline|candidate=mul32|baseline-median|mul32-median|mul32-vs-baseline|production=|GPU total|throughput|Best batch|FAILED|Fatal|CUDA error' \
"${PRIME_PREFIX}"-gpu*.log || true

echo

if grep -Eq 'baseline-parity=FAIL|mul32-parity=FAIL|FAILED|Fatal|CUDA error' \
   "${PRIME_PREFIX}"-gpu*.log; then
    echo "Candidate failed correctness validation."
    echo "The foreground run will use the parity-safe baseline selected by the cache."
    CANDIDATE_OK=0
else
    CANDIDATE_OK="$(python3 - "$PRIME_PREFIX" <<'PY'
import pathlib, re, sys
prefix = pathlib.Path(sys.argv[1])
ok = True
for gpu in (0, 1):
    path = pathlib.Path(f"{prefix}-gpu{gpu}.log")
    text = path.read_text(errors="replace")
    values = [float(x) for x in re.findall(r"mul32-vs-baseline=([+-]?[0-9.]+)%", text)]
    promoted = "production=4-lane-ttable-cg" in text
    gain = values[-1] if values else float("-inf")
    print(
        f"GPU {gpu}: split-multiply gain={gain:+.3f}% | "
        f"promoted={'yes' if promoted else 'no'}",
        file=sys.stderr,
    )
    if not promoted or gain < 8.0:
        ok = False
print("1" if ok else "0")
PY
)"
fi

# Nothing from the forced benchmark may leak into the production run.
unset YERBAS_BENCH_FORCE_CN_FAST || true
unset YERBAS_BENCH_FORCE_CN_VARIANT || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_DIAGNOSTICS || true
unset YERBAS_CUDA_PROFILE || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true
unset YERBAS_GPU_TUNE_MODE || true

RUN_BIN="$BUILD_DIR/yerbas-miner"
RUN_XDG_ROOT="$WORK_XDG_ROOT"

if [[ "$CANDIDATE_OK" == "1" ]]; then
    RUN_NAME="4lane-splitmul-clean-workday"
    EXPECTED_MODE="4-lane-ttable-cg on both GPUs"
    echo
    echo "GO: split-multiply cleared >=8% on both GPUs."
    echo "The workday run will exercise the promoted candidate."
else
    RUN_NAME="baseline-clean-workday"
    EXPECTED_MODE="4-lane-ttable baseline"
    echo
    echo "NO-GO: split-multiply missed the >=8%/both-GPU bar."
    echo "The day will still produce a valuable clean pool-side BASELINE using"
    echo "the same freshly completed CPU/GPU tuning profile, without retuning."
fi

LOG="$LOG_DIR/${RUN_NAME}-$(date +%Y%m%d-%H%M%S).log"

printf '\n%s\n' "============================================================"
echo " YERBAS MINER - FOREGROUND WORKDAY RUN"
echo " Binary:        $RUN_BIN"
echo " Cache root:    $RUN_XDG_ROOT"
echo " Expected mode: $EXPECTED_MODE"
echo " Log:           $LOG"
echo ""
echo " This process is ATTACHED to this terminal."
echo " Leave the terminal open. Press Ctrl+C to stop."
echo " No first-run autotune prompt should appear."
printf '%s\n\n' "============================================================"

# The benchmark above has already written the CN-Fast decision into the cloned
# cache. Normal auto/cache-first production policy is used from this point on.
set +e
XDG_CACHE_HOME="$RUN_XDG_ROOT" \
script -q -f -e -c "$RUN_BIN" "$LOG"
STATUS=$?
set -e

echo
echo "================ POST-RUN CHECK ================"
grep -E \
'CryptoNight production selector cache loaded \| CN-Fast|AVG[[:space:]]|TOTAL[[:space:]].*ACCEPTED|Fatal|CUDA error|FAILED' \
"$LOG" | tail -30 || true
echo "Log: $LOG"
exit "$STATUS"
