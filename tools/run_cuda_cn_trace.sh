#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cmake -S . -B build-cuda
python3 tools/trace_core_cn.py build-cuda/_deps/yerbas_core-src/src/cryptonote/slow-hash.c
cmake --build build-cuda --target cuda-keccak-validation --parallel "$(nproc)"
./build-cuda/cuda-keccak-validation
