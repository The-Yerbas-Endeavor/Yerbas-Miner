#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-cuda}"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DYERBAS_NATIVE_CPU=ON

cmake --build "${BUILD_DIR}" \
  --target yerbas-miner \
  --parallel "$(nproc)"

echo
echo "Native CPU-tuned Yerbas Miner built in ${BUILD_DIR}."
echo "This binary is tuned for this host CPU; do not redistribute it as a portable build."
