#include "miner.h"

// The CUDA backend remains one translation unit while the stable implementation
// is being modularized. Production tuning is intentionally simple: base hash
// kernels + one whole-GhostRider GPU policy tuner. Retired batch/CN/cooperative
// micro-tuners are no longer compiled into the production backend.
#include "cuda/generated/cuda_backend_part1.inc"
#include "cuda/generated/cuda_backend_part2a.inc"

// Keep the proven split-CryptoNight implementation as the untuned hash engine.
// Its historical autotune entry points are renamed so they cannot be selected by
// production. The new GPU policy supplies geometry directly.
#define autotune_cn_geometries autotune_cn_geometries_legacy
#define launch_split_cryptonight_variant launch_split_cryptonight_variant_legacy
#define launch_split_cryptonight launch_split_cryptonight_legacy
#include "cuda/generated/cuda_backend_part2b.inc"
#undef launch_split_cryptonight
#undef launch_split_cryptonight_variant
#undef autotune_cn_geometries

#include "cuda/generated/cuda_backend_gpu_policy.inc"
#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
