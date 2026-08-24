#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-release}"

cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DYERBAS_ENABLE_CUDA=ON \
  -DYERBAS_NATIVE_CPU=OFF \
  -DYERBAS_CUDA_ARCHITECTURES="52;60;61;70;75;80;86;89;90"

cmake --build "${BUILD_DIR}" \
  --target yerbas-miner \
  --parallel "$(nproc)"

echo
echo "Generic portable Yerbas Miner built in ${BUILD_DIR}."
echo "CPU code is not compiled with -march=native/-mtune=native."
echo "CUDA fat binary includes supported NVIDIA architectures: 52,60,61,70,75,80,86,89,90."
echo "Runtime CPU/GPU autotuning will select production settings on the target machine."
