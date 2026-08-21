#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
CORE_DIR="$BUILD_DIR/_deps/yerbas_core-src"
SLOW_HASH="$CORE_DIR/src/cryptonote/slow-hash.c"

# Configure only when the build tree does not exist. A normal `cmake --build`
# will automatically re-run CMake if CMakeLists.txt changed, so avoiding an
# unconditional configure here keeps generated CUDA headers from getting new
# timestamps on every trace cycle and prevents needless recompilation of
# cuda_backend.cu / sph_batch_validation.cu.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR"
fi

# The trace patch edits only the fetched pinned Core copy. Restore that file
# directly instead of reconfiguring the entire project every run.
if [[ -d "$CORE_DIR/.git" || -f "$CORE_DIR/.git" ]]; then
  git -C "$CORE_DIR" checkout -- src/cryptonote/slow-hash.c
else
  echo "Missing fetched Yerbas Core checkout; configuring build tree..."
  cmake -S . -B "$BUILD_DIR"
  git -C "$CORE_DIR" checkout -- src/cryptonote/slow-hash.c
fi

python3 tools/trace_core_cn.py "$SLOW_HASH"

# Ninja handles file-level parallelism with $(nproc); NVCC itself is also
# configured with --threads=0 and, on CUDA 12+, --split-compile=0.
cmake --build "$BUILD_DIR" \
  --target cuda-keccak-validation \
  --parallel "$(nproc)"

./"$BUILD_DIR"/cuda-keccak-validation
