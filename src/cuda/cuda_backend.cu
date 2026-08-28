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

// Keep the original cooperative setup/loop/final kernels compiled as a parity
// reference. Production setup/final use the compact shared-key variants below.
#include "cuda/generated/cuda_backend_cn_cooperative.inc"
#include "cuda/generated/cuda_backend_cn_sharedkey.inc"

// Two-lane memory-loop candidate. This file also owns the one-time direct parity
// gate against the current four-lane production reference.
#include "cuda/generated/cuda_backend_cn_2lane.inc"

// Sparse production diagnostics around the exact same production kernels.
// Rename setup/final symbols consumed by the wrapper to the compact shared-key
// variants; the loop is selected by the direct parity gate inside the wrapper.
#define cryptonight_setup_stage_cooperative8 cryptonight_setup_stage_cooperative8_sharedkey
#define cryptonight_final_stage_cooperative8 cryptonight_final_stage_cooperative8_sharedkey
#include "cuda/generated/cuda_backend_cn_phase_profile.inc"
#undef cryptonight_final_stage_cooperative8
#undef cryptonight_setup_stage_cooperative8

// Keep the bounded scratchpad-class tuner helpers, but replace its public policy
// entry point with the hardened wrapper below.  The wrapper verifies the actual
// scratchpad requirement of every representative GhostRider rotation before a
// calibration launch, so a missing light class can never run a heavy CN job at
// an unsafe batch size.
#define build_gpu_tune_policy build_gpu_tune_policy_unchecked
#include "cuda/generated/cuda_backend_gpu_faststart.inc"
#undef build_gpu_tune_policy
#include "cuda/generated/cuda_backend_gpu_calibration_safe.inc"

// Route production scans through the phase-profile wrapper. Setup/final use the
// shared-key 8-lane path; the memory loop promotes from 4 lanes to 2 only after
// exact per-GPU/per-variant output parity succeeds.
#define launch_split_cryptonight launch_split_cryptonight_phase_profiled
#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
#undef launch_split_cryptonight
