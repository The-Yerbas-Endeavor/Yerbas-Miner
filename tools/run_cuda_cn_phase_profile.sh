#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DYERBAS_ENABLE_CUDA=ON
fi

NVCC="${CUDACXX:-$(command -v nvcc)}"
OUT="$BUILD_DIR/cuda-cn-phase-profile"

"$NVCC" \
  -std=c++17 -O3 -DNDEBUG -arch=native --threads=0 \
  -I"$PWD/src" \
  -I"$PWD/$BUILD_DIR/generated_cuda" \
  "$PWD/tools/cuda_cn_phase_profile.cu" \
  -o "$OUT"

exec "$OUT" "$@"
