#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANDIDATE_REF="origin/feature/cn-4lane-splitmul"
STABLE_REF="origin/stable/overnight-2026-09-15"
BUILD_DIR="$ROOT/build-4lane-splitmul"
STABLE_BUILD="$ROOT/build-stable"
LOG_DIR="$ROOT/logs"

# Preserve the user's normal production tuning cache. The experimental run gets
# a private clone so selector experiments cannot contaminate production state.
PROD_XDG_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
PROD_CACHE="$PROD_XDG_ROOT/yerbas-miner"
WORK_XDG_ROOT="$ROOT/.bench-cache/workday-4lane-splitmul"
WORK_CACHE="$WORK_XDG_ROOT/yerbas-miner"

mkdir -p "$LOG_DIR"

git fetch origin

if [[ ! -d "$PROD_CACHE" ]]; then
    echo "ERROR: production tuning cache not found: $PROD_CACHE"
    echo "Run the known-good miner once with gpu_tune=auto so the hardware profile exists."
    exit 1
fi

# Refuse to create another accidental first-run session. These globs are the
# current production hardware/class and per-variant tuning families.
if ! compgen -G "$PROD_CACHE/cpu-policy-v6*" >/dev/null || \
   ! compgen -G "$PROD_CACHE/gpu-calibration-v2*" >/dev/null || \
   ! compgen -G "$PROD_CACHE/gpu-variant-batch-v2*" >/dev/null; then
    echo "ERROR: production cache exists but the completed tuning profile is incomplete."
    echo "Expected cpu-policy-v6, gpu-calibration-v2 and gpu-variant-batch-v2 entries."
    exit 1
fi

echo
printf '%s\n' "============================================================" \
              " CLEAN WORKDAY CANDIDATE TEST" \
              " - clones existing production tuning" \
              " - retests only CN-Fast phase-2 selector" \
              " - requires >=8% candidate gain on BOTH GPUs" \
              " - miner remains attached in foreground" \
              "============================================================"
echo

git switch --detach "$CANDIDATE_REF"

# Incremental build. No clean rebuild unless the caller removed the directory.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi
cmake --build "$BUILD_DIR" --target cuda-batch-benchmark yerbas-miner -j"$(nproc)"

# Clone all known CPU/GPU tuning state first. This is what prevents the first-run
# autotune question while keeping the production cache itself read-only.
rm -rf "$WORK_XDG_ROOT"
mkdir -p "$WORK_XDG_ROOT"
cp -a "$PROD_CACHE" "$WORK_CACHE"

# Variant index 2 is CN-Fast. Invalidate only decisions whose implementation is
# changed by this branch. Hardware calibration, per-variant batch sizing, setup/
# final backends and all unrelated CN variants remain warm.
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
    echo "Candidate failed correctness validation; stable miner will be used."
    CANDIDATE_OK=0
else
    CANDIDATE_OK="$(python3 - "$PRIME_PREFIX" <<'PY'
import pathlib, re, sys
prefix = pathlib.Path(sys.argv[1])
ok = True
for gpu in (0, 1):
    p = pathlib.Path(f"{prefix}-gpu{gpu}.log")
    text = p.read_text(errors="replace")
    vals = [float(x) for x in re.findall(r"mul32-vs-baseline=([+-]?[0-9.]+)%", text)]
    promoted = "production=4-lane-ttable-cg" in text
    gain = vals[-1] if vals else float("-inf")
    print(f"GPU {gpu}: split-multiply gain={gain:+.3f}% | promoted={'yes' if promoted else 'no'}", file=sys.stderr)
    if not promoted or gain < 8.0:
        ok = False
print("1" if ok else "0")
PY
)"
fi

# No developer tuning controls are allowed to leak into the workday miner.
unset YERBAS_BENCH_FORCE_CN_FAST || true
unset YERBAS_BENCH_FORCE_CN_VARIANT || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_DIAGNOSTICS || true
unset YERBAS_CUDA_PROFILE || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true
unset YERBAS_GPU_TUNE_MODE || true

if [[ "$CANDIDATE_OK" == "1" ]]; then
    RUN_BIN="$BUILD_DIR/yerbas-miner"
    RUN_XDG_ROOT="$WORK_XDG_ROOT"
    RUN_NAME="4lane-splitmul-clean-workday"
    EXPECTED_MODE="4-lane-ttable-cg"
    echo
    echo "GO: candidate cleared >=8% on both GPUs."
    echo "Running candidate with cloned warm tuning profile."
else
    echo
    echo "NO-GO: candidate missed the >=8%/both-GPU bar."
    echo "Running the stable reference instead; no workday is wasted on a weak kernel."
    git switch --detach "$STABLE_REF"
    if [[ ! -x "$STABLE_BUILD/yerbas-miner" ]]; then
        rm -rf "$STABLE_BUILD"
        ./scripts/configure-linux-cuda.sh "$STABLE_BUILD"
        cmake --build "$STABLE_BUILD" --target yerbas-miner -j"$(nproc)"
    fi
    RUN_BIN="$STABLE_BUILD/yerbas-miner"
    RUN_XDG_ROOT="$PROD_XDG_ROOT"
    RUN_NAME="stable-clean-workday"
    EXPECTED_MODE="stable-reference"
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

# Keep overlap at its ordinary cache-first policy for the actual production run.
# The CN-Fast kernel selector itself has already been decided above.
XDG_CACHE_HOME="$RUN_XDG_ROOT" \
script -q -f -e -c "$RUN_BIN" "$LOG"

STATUS=$?

echo
echo "================ POST-RUN CHECK ================"ngrep -E 'CryptoNight production selector cache loaded \| CN-Fast|AVG[[:space:]]|TOTAL[[:space:]].*ACCEPTED|Fatal|CUDA error|FAILED' "$LOG" | tail -30 || true
echo "Log: $LOG"
exit "$STATUS"
