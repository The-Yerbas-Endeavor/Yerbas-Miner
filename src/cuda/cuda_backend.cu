#include "miner.h"

// The CUDA backend remains one translation unit while the stable implementation
// is being modularized. Production tuning is intentionally simple: base hash
// kernels + one whole-GhostRider GPU policy tuner.
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
// This mirrors the proven CUDA architecture used by high-performance CN miners:
// distribute one hash across a small sub-warp instead of carrying the complete
// dependency chain in one CUDA lane.  It is generic and contains no device-name
// or compute-capability selection rules.
#include "cuda/generated/cuda_backend_cn_cooperative.inc"

// The unified policy wrapper expects the historical production entry-point
// name. Redirect that one call to the clean cooperative implementation while
// retaining the legacy scalar implementation as a reference in this TU.
#define launch_split_cryptonight_variant_legacy launch_split_cryptonight_variant_production
#include "cuda/generated/cuda_backend_gpu_policy.inc"
#undef launch_split_cryptonight_variant_legacy

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
