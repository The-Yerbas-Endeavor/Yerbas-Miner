#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
GPU0="${1:-0}"
GPU1="${2:-1}"
BATCH="${3:-3584}"

if [[ "$GPU0" == "$GPU1" ]]; then
  echo "GPU ids must be different" >&2
  exit 2
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DYERBAS_ENABLE_CUDA=ON
fi

cmake --build "$BUILD_DIR" \
  --target cuda-batch-benchmark \
  --parallel "$(nproc)"

BENCH="./$BUILD_DIR/cuda-batch-benchmark"
TMPDIR_PROFILE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_PROFILE"' EXIT

extract_hps() {
  awk '/Best batch:/ {print $(NF-1)}' "$1" | tail -n1
}

print_result() {
  local label="$1"
  local logfile="$2"
  local hps
  hps="$(extract_hps "$logfile")"
  if [[ -z "$hps" ]]; then
    echo "$label: FAILED"
    cat "$logfile"
    return 1
  fi
  printf '%-24s %8.2f H/s\n' "$label" "$hps"
}

echo "Yerbas CUDA multi-GPU production-pipeline profile"
echo "GPU 0 id: $GPU0 | GPU 1 id: $GPU1 | batch: $BATCH"
echo

echo "[1/3] Solo GPU $GPU0..."
"$BENCH" "$GPU0" "$BATCH" >"$TMPDIR_PROFILE/solo0.log" 2>&1
SOLO0="$(extract_hps "$TMPDIR_PROFILE/solo0.log")"
print_result "Solo GPU $GPU0" "$TMPDIR_PROFILE/solo0.log"

echo "[2/3] Solo GPU $GPU1..."
"$BENCH" "$GPU1" "$BATCH" >"$TMPDIR_PROFILE/solo1.log" 2>&1
SOLO1="$(extract_hps "$TMPDIR_PROFILE/solo1.log")"
print_result "Solo GPU $GPU1" "$TMPDIR_PROFILE/solo1.log"

echo "[3/3] GPU $GPU0 + GPU $GPU1 concurrently..."
START_NS="$(date +%s%N)"
"$BENCH" "$GPU0" "$BATCH" >"$TMPDIR_PROFILE/dual0.log" 2>&1 &
PID0=$!
"$BENCH" "$GPU1" "$BATCH" >"$TMPDIR_PROFILE/dual1.log" 2>&1 &
PID1=$!
RC0=0
RC1=0
wait "$PID0" || RC0=$?
wait "$PID1" || RC1=$?
STOP_NS="$(date +%s%N)"

if (( RC0 != 0 || RC1 != 0 )); then
  echo "Concurrent profile failed: GPU $GPU0 rc=$RC0, GPU $GPU1 rc=$RC1" >&2
  echo "--- GPU $GPU0 output ---" >&2
  cat "$TMPDIR_PROFILE/dual0.log" >&2
  echo "--- GPU $GPU1 output ---" >&2
  cat "$TMPDIR_PROFILE/dual1.log" >&2
  exit 3
fi

DUAL0="$(extract_hps "$TMPDIR_PROFILE/dual0.log")"
DUAL1="$(extract_hps "$TMPDIR_PROFILE/dual1.log")"
WALL_MS="$(( (STOP_NS - START_NS) / 1000000 ))"

print_result "Concurrent GPU $GPU0" "$TMPDIR_PROFILE/dual0.log"
print_result "Concurrent GPU $GPU1" "$TMPDIR_PROFILE/dual1.log"

python3 - "$SOLO0" "$SOLO1" "$DUAL0" "$DUAL1" "$WALL_MS" <<'PY'
import sys
s0, s1, d0, d1 = map(float, sys.argv[1:5])
wall_ms = int(sys.argv[5])
solo_sum = s0 + s1
dual_sum = d0 + d1
retention = (dual_sum / solo_sum * 100.0) if solo_sum else 0.0
loss = solo_sum - dual_sum
print()
print("================ MULTI-GPU SUMMARY ================")
print(f"Solo sum:             {solo_sum:8.2f} H/s")
print(f"Concurrent sum:       {dual_sum:8.2f} H/s")
print(f"Concurrency retention:{retention:8.2f}%")
print(f"Concurrent loss:      {loss:8.2f} H/s")
print(f"Pair wall time:       {wall_ms/1000.0:8.2f} s")
if retention >= 95.0:
    print("Diagnosis: little multi-GPU contention; optimize miner scheduling next.")
elif retention >= 85.0:
    print("Diagnosis: moderate contention; test staggered launches/per-GPU batch sizes next.")
else:
    print("Diagnosis: heavy contention; inspect host/CUDA scheduling and memory pressure first.")
print("===================================================")
PY

echo
echo "Raw logs were temporary; rerun with:"
echo "  bash tools/run_cuda_multigpu_profile.sh $GPU0 $GPU1 $BATCH"
