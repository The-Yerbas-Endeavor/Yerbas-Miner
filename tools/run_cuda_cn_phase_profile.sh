#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DYERBAS_ENABLE_CUDA=ON
fi

NVCC="${CUDACXX:-$(command -v nvcc)}"
OUT="$BUILD_DIR/cuda-cn-phase-profile"
CORE_SRC="$BUILD_DIR/_deps/yerbas_core-src/src"

if [[ ! -f "$CORE_SRC/crypto/sph_types.h" ]]; then
  echo "Missing pinned Yerbas Core headers under $CORE_SRC" >&2
  echo "Re-run CMake configure for build-cuda first." >&2
  exit 2
fi

"$NVCC" \
  -std=c++17 -O3 -DNDEBUG -arch=native --threads=0 \
  -I"$PWD/src" \
  -I"$PWD/$BUILD_DIR/generated_cuda" \
  -I"$PWD/$CORE_SRC/crypto" \
  -I"$PWD/$CORE_SRC/cryptonote" \
  "$PWD/tools/cuda_cn_phase_profile.cu" \
  -o "$OUT"

exec "$OUT" "$@"
