#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${YERBAS_BUILD_DIR:-$ROOT/build-production-latency}"
BIN="$BUILD_DIR/cuda-batch-benchmark"
GPU="${YERBAS_BENCH_GPU:-0}"
VARIANT="${YERBAS_BENCH_VARIANT:-fast}"
OUT="$ROOT/logs/nvprof-kernel-discovery-gpu${GPU}-${VARIANT}-$(date +%Y%m%d-%H%M%S).log"

NVPROF_BIN="$(command -v nvprof 2>/dev/null || true)"
if [[ -z "$NVPROF_BIN" ]]; then
    NVPROF_BIN="$(find /usr/local -maxdepth 3 -type f -name nvprof -path '*/cuda*/bin/nvprof' 2>/dev/null | sort -V | tail -1 || true)"
fi
if [[ -z "$NVPROF_BIN" ]]; then
    echo "ERROR: nvprof not found." >&2
    exit 2
fi

cmake --build "$BUILD_DIR" --target cuda-batch-benchmark -j"$(nproc)"

echo "============================================================"
echo " YERBAS NVPROF KERNEL DISCOVERY"
echo "============================================================"
echo " GPU:      $GPU"
echo " Variant:  $VARIANT"
echo " Metric:   NONE (trace-only discovery)"
echo " Filter:   NONE"
echo " Log:      $OUT"
echo "============================================================"

sudo env \
    "HOME=$HOME" \
    "XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}" \
    "YERBAS_CUDA_BENCH_RAW_BATCH=1" \
    "YERBAS_BENCH_FORCE_CN_VARIANT=$VARIANT" \
    "$NVPROF_BIN" \
    --devices "$GPU" \
    --profile-api-trace none \
    --print-gpu-trace \
    --log-file "$OUT" \
    "$BIN" "$GPU" 3584

sudo chown "$(id -u):$(id -g)" "$OUT" 2>/dev/null || true

echo
echo "Kernel-name matches containing CryptoNight/ttable:"
grep -Ei 'cryptonight|ttable' "$OUT" | head -120 || true

echo
echo "Saved: $OUT"
