#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR"
fi

cmake --build "$BUILD_DIR" \
  --target cuda-batch-benchmark \
  --parallel "$(nproc)"

./"$BUILD_DIR"/cuda-batch-benchmark "$@"
