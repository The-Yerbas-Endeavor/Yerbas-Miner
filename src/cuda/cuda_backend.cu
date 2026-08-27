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

// Production CryptoNight uses cooperative setup/loop/final kernels. It is
// generic CUDA and contains no device-name or compute-capability rules.
#include "cuda/generated/cuda_backend_cn_cooperative.inc"

// Sparse production diagnostics around the exact same cooperative kernels.
// Normal launches do not create events; YERBAS_DIAGNOSTICS=1 samples setup,
// loop and final separately so the next optimization is data-driven.
#include "cuda/generated/cuda_backend_cn_phase_profile.inc"

// Keep the bounded scratchpad-class tuner helpers, but replace its public policy
// entry point with the hardened wrapper below.  The wrapper verifies the actual
// scratchpad requirement of every representative GhostRider rotation before a
// calibration launch, so a missing light class can never run a heavy CN job at
// an unsafe batch size.
#define build_gpu_tune_policy build_gpu_tune_policy_unchecked
#include "cuda/generated/cuda_backend_gpu_faststart.inc"
#undef build_gpu_tune_policy
#include "cuda/generated/cuda_backend_gpu_calibration_safe.inc"

// Route production scans through the phase-profile wrapper. With diagnostics
// disabled this is the same 8-lane setup / 4-lane loop / 8-lane final path.
#define launch_split_cryptonight launch_split_cryptonight_phase_profiled
#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
#undef launch_split_cryptonight
