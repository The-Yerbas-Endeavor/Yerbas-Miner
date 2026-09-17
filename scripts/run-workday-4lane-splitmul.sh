#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANDIDATE_REF="origin/feature/cn-4lane-splitmul"
BUILD_DIR="$ROOT/build-4lane-splitmul"
LOG_DIR="$ROOT/logs"

# Prefer the completed profile produced by the previous workday first-run tune.
# Fall back to the user's normal production cache if that profile is unavailable.
NORMAL_XDG_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
NORMAL_CACHE="$NORMAL_XDG_ROOT/yerbas-miner"
PREVIOUS_TUNED_CACHE="$ROOT/.bench-cache/4lane-splitmul/yerbas-miner"
WORK_XDG_ROOT="$ROOT/.bench-cache/workday-4lane-splitmul"
WORK_CACHE="$WORK_XDG_ROOT/yerbas-miner"

# Match the exact prefixes used by src/first_run.h. These are on-disk filename
# prefixes, not the human-readable cache labels printed by the tuners.
has_required_profile() {
    local cache="$1"
    [[ -d "$cache" ]] && \
    compgen -G "$cache/cpu-policy-rev*" >/dev/null && \
    compgen -G "$cache/gpu-calibration-rev*" >/dev/null
}

show_profile_files() {
    local cache="$1"
    echo "Profile files in $cache:"
    find "$cache" -maxdepth 1 -type f \
      \( -name 'cpu-policy-rev*' -o \
         -name 'gpu-calibration-rev*' -o \
         -name 'gpu-variant-batch-rev*' \) \
      -printf '  %f\n' 2>/dev/null | sort || true
}

mkdir -p "$LOG_DIR"
git fetch origin

if has_required_profile "$PREVIOUS_TUNED_CACHE"; then
    SOURCE_CACHE="$PREVIOUS_TUNED_CACHE"
    SOURCE_LABEL="previous completed workday autotune"
elif has_required_profile "$NORMAL_CACHE"; then
    SOURCE_CACHE="$NORMAL_CACHE"
    SOURCE_LABEL="normal production tuning cache"
else
    echo "ERROR: no first-run-safe tuning profile was found."
    echo "The miner itself looks for cpu-policy-rev* and gpu-calibration-rev*."
    echo
    echo "Checked:"
    echo "  $PREVIOUS_TUNED_CACHE"
    echo "  $NORMAL_CACHE"
    echo
    for cache in "$PREVIOUS_TUNED_CACHE" "$NORMAL_CACHE"; do
        if [[ -d "$cache" ]]; then
            echo "Files currently present in $cache:"
            find "$cache" -maxdepth 1 -type f -printf '  %f\n' 2>/dev/null | sort | head -80 || true
            echo
        fi
    done
    echo "Refusing to start anything that could stop at the interactive autotune prompt."
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
show_profile_files "$SOURCE_CACHE"
echo

git switch --detach "$CANDIDATE_REF"

# Incremental build; avoid another unnecessary clean CUDA rebuild.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ./scripts/configure-linux-cuda.sh "$BUILD_DIR"
fi
cmake --build "$BUILD_DIR" --target cuda-batch-benchmark yerbas-miner -j"$(nproc)"

# Clone the completed tuning profile. The original production/tuned cache is
# never modified by this experiment.
rm -rf "$WORK_XDG_ROOT"
mkdir -p "$WORK_XDG_ROOT"
cp -a "$SOURCE_CACHE" "$WORK_CACHE"

if ! has_required_profile "$WORK_CACHE"; then
    echo "ERROR: cloned work cache lost the first-run tuning profile."
    exit 1
fi

# Variant index 2 is CN-Fast. Invalidate only selector/geometry decisions whose
# implementation is changed by this branch. CPU policy, GPU hardware calibration,
# per-variant batch sizing, setup/final backends, and all unrelated CN variants
# remain warm.
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
        f"built-in promoted={'yes' if promoted else 'no'}",
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
    # Our production bar is 8%, stricter than the miner's built-in 5% selector.
    # If a 5-8% result was promoted internally, force this cloned cache back to
    # the proven mode-44 baseline before the all-day run.
    python3 - "$WORK_CACHE" "$PRIME_PREFIX" <<'PY'
import pathlib, re, sys
cache = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]

for gpu in (0, 1):
    text = pathlib.Path(f"{prefix}-gpu{gpu}.log").read_text(errors="replace")
    m = re.findall(r"baseline-median=[0-9.]+ ms@([0-9]+)", text)
    if not m:
        raise SystemExit(f"ERROR: could not determine baseline threads for GPU {gpu}")
    threads = int(m[-1])

    matches = sorted(cache.glob(f"cn-production-rev2-v2-*-dev{gpu}-*-batch3584-*.txt"))
    if not matches:
        # Keep this fallback for cache-key formatting changes while still
        # requiring a unique GPU/batch selector file.
        matches = [
            p for p in cache.glob("cn-production-rev2-v2-*.txt")
            if f"-dev{gpu}-" in p.name and "-batch3584-" in p.name
        ]
    if len(matches) != 1:
        raise SystemExit(
            f"ERROR: expected one CN-Fast selector cache for GPU {gpu}, found {len(matches)}"
        )

    path = matches[0]
    parts = path.read_text().split()
    if len(parts) < 5 or parts[0] != "YERBAS_CN_PRODUCTION" or parts[1] != "2" or parts[2] != "2":
        raise SystemExit(f"ERROR: unexpected selector cache format: {path}")
    parts[3] = "44"
    parts[4] = str(threads)
    path.write_text(" ".join(parts[:5]) + "\n")
    print(f"GPU {gpu}: locked CN-Fast workday cache to mode 44 @ {threads} threads")
PY

    RUN_NAME="baseline-clean-workday"
    EXPECTED_MODE="4-lane-ttable baseline"
    echo
    echo "NO-GO: split-multiply missed the >=8%/both-GPU bar."
    echo "The all-day run is explicitly locked to the proven mode-44 baseline."
fi

# Last safeguard: these exact profile prefixes are what first_run.h checks. If
# either disappears, abort here instead of ever showing an interactive prompt.
if ! has_required_profile "$WORK_CACHE"; then
    echo "ERROR: work cache is no longer first-run safe. Aborting."
    exit 1
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

# The benchmark above wrote the CN-Fast decision into the cloned cache. Normal
# auto/cache-first production policy is used from this point on.
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
