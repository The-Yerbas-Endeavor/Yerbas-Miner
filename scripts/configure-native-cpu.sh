#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-native}"

cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DYERBAS_ENABLE_CUDA=ON \
  -DYERBAS_NATIVE_CPU=ON \
  -DYERBAS_CUDA_ARCHITECTURES=native

cmake --build "${BUILD_DIR}" \
  --target yerbas-miner \
  --parallel "$(nproc)"

echo
echo "Native benchmark Yerbas Miner built in ${BUILD_DIR}."
echo "WARNING: CPU code is tuned for this host CPU with native compiler optimizations."
echo "WARNING: CUDA is also built for the locally detected GPU architecture."
echo "Do not redistribute this binary as the generic Yerbas Miner release."
echo "Use scripts/configure-release.sh for portable development, testing, and releases."
