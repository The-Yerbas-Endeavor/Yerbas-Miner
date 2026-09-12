#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$repo_root/build}"

# Do not rely on CMake's CUDA_ARCHITECTURES=native probe on headless/mining
# systems. Build a broad, architecture-generic binary instead. compute_52 PTX
# provides a forward-compatible fallback while common generations get native
# cubins. No GPU model names or device-specific tuning are used here.
architectures='52-virtual;61-real;75-real;86-real;89-real;90-real'

echo "Configuring Yerbas Miner Linux CUDA build"
echo "  source: $repo_root"
echo "  build:  $build_dir"
echo "  CUDA architectures: $architectures"

cmake -S "$repo_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DYERBAS_ENABLE_CUDA=ON \
  -DBUILD_TESTING=OFF \
  "-DYERBAS_CUDA_ARCHITECTURES=$architectures"

echo
echo "Configuration complete. Build with:"
echo "  cmake --build \"$build_dir\" --target yerbas-miner -j\"\$(nproc)\""
