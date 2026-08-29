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

#include "cuda/generated/cuda_backend_cn_cooperative.inc"
#include "cuda/generated/cuda_backend_cn_sharedkey.inc"
#include "cuda/generated/cuda_backend_cn_2lane.inc"
#include "cuda/generated/cuda_backend_cn_ttable_loop.inc"
#include "cuda/generated/cuda_backend_cn_tile64.inc"
#include "cuda/generated/cuda_backend_cn_cg_loop.inc"
#include "cuda/generated/cuda_backend_cn_residency.inc"
#include "cuda/generated/cuda_backend_cn_selector_hardened.inc"

#define cryptonight_setup_stage_cooperative8 cryptonight_setup_stage_cooperative8_sharedkey
#define cryptonight_final_stage_cooperative8 cryptonight_final_stage_cooperative8_sharedkey
#include "cuda/generated/cuda_backend_cn_phase_profile.inc"
#undef cryptonight_final_stage_cooperative8
#undef cryptonight_setup_stage_cooperative8

#define build_gpu_tune_policy build_gpu_tune_policy_unchecked
#include "cuda/generated/cuda_backend_gpu_faststart.inc"
#undef build_gpu_tune_policy
#include "cuda/generated/cuda_backend_gpu_calibration_safe.inc"

#define launch_split_cryptonight launch_split_cryptonight_phase_profiled
#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
#undef launch_split_cryptonight
