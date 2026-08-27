#include "miner.h"

// The CUDA backend remains one translation unit while the stable implementation
// is being modularized. Production startup is intentionally non-blocking:
// base hash kernels + one runtime-safe instant-start GPU policy.
#include "cuda/generated/cuda_backend_part1.inc"
#include "cuda/generated/cuda_backend_part2a.inc"

// Keep the historical split-CryptoNight implementation available as a known
// scalar reference, but do not dispatch it directly in production.
#define autotune_cn_geometries autotune_cn_geometries_legacy
#define launch_split_cryptonight_variant launch_split_cryptonight_variant_legacy
#define launch_split_cryptonight launch_split_cryptonight_legacy
#include "cuda/generated/cuda_backend_part2b.inc"
#undef launch_split_cryptonight
#undef launch_split_cryptonight_variant
#undef autotune_cn_geometries

// Production CryptoNight uses one clean four-lane cooperative memory loop.
// It is generic CUDA and contains no device-name or compute-capability rules.
#include "cuda/generated/cuda_backend_cn_cooperative.inc"

// Production must begin mining immediately.  The old synthetic whole-GhostRider
// startup benchmark remains out of the production path; optimization is driven
// by live rotation data instead.
#include "cuda/generated/cuda_backend_gpu_faststart.inc"

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
