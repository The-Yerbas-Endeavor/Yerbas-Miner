#include "miner.h"

#include <mutex>

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
#include "cuda/generated/cuda_backend_cn_wordphase.inc"
#include "cuda/generated/cuda_backend_cn_wordphase_selector.inc"
#include "cuda/generated/cuda_backend_cn_2lane.inc"
#include "cuda/generated/cuda_backend_cn_ttable_loop.inc"
#include "cuda/generated/cuda_backend_cn_tile64.inc"
#include "cuda/generated/cuda_backend_cn_cg_loop.inc"
#include "cuda/generated/cuda_backend_cn_residency.inc"
#include "cuda/generated/cuda_backend_cn_selector_hardened.inc"
#include "cuda/generated/cuda_backend_cn_blocksize_tuner.inc"

#define cryptonight_setup_stage_cooperative8 cryptonight_setup_stage_cooperative8_sharedkey
#define cryptonight_final_stage_cooperative8 cryptonight_final_stage_cooperative8_sharedkey
// Retain the previous profiler/scheduler as the safe fallback, but rename its
// public dispatch points so the staggered scheduler can own production routing.
#define launch_cn_scheduled_if_ready launch_cn_scheduled_if_ready_legacy
#define launch_split_cryptonight_variant_phase_profiled launch_split_cryptonight_variant_phase_profiled_legacy
#define launch_split_cryptonight_phase_profiled launch_split_cryptonight_phase_profiled_legacy
#include "cuda/generated/cuda_backend_cn_phase_profile.inc"
#undef launch_split_cryptonight_phase_profiled
#undef launch_split_cryptonight_variant_phase_profiled
#undef launch_cn_scheduled_if_ready

// The staggered scheduler's setup/loop/final launchers are selector-aware and
// lazily initialize their own parity-tested production choices. Requiring those
// selectors to be initialized before entering the staggered path creates a
// circular gate: the code that initializes them can never run. Keep the legacy
// scheduler's conservative readiness check unchanged, but let staggered dispatch
// enter whenever the device index is valid; its normal launchers still own all
// parity and tuning decisions before any candidate is promoted.
template <std::uint8_t VariantIndex>
bool cn_stagger_selectors_ready(int device_id, bool)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    return device_id >= 0 && device_id < kCnPhaseProfileMaxDevices;
}

// Capture the CUDA runtime entry point before temporarily overriding the public
// cudaGetDeviceProperties macro below. Newer CUDA headers define that name as a
// versioned macro alias, so restoring it through another temporary macro leaves a
// dangling token once the helper macro is undefined.
cudaError_t yerbas_cuda_get_device_properties(cudaDeviceProp* props, int device_id)
{
    return cudaGetDeviceProperties(props, device_id);
}

// cudaDeviceProp is immutable for the lifetime of the process, but the stagger
// wrapper previously queried it for every CryptoNight stage. Cache one copy per
// device and preserve the CUDA error contract expected by check_cuda().
cudaError_t cn_stagger_cached_device_properties(cudaDeviceProp* props, int device_id)
{
    if (props == nullptr) return cudaErrorInvalidValue;
    if (device_id < 0 || device_id >= kCnPhaseProfileMaxDevices)
        return yerbas_cuda_get_device_properties(props, device_id);

    static std::array<std::once_flag, kCnPhaseProfileMaxDevices> once{};
    static std::array<cudaDeviceProp, kCnPhaseProfileMaxDevices> cached{};
    static std::array<cudaError_t, kCnPhaseProfileMaxDevices> result{};

    std::call_once(once[device_id], [device_id]() {
        result[device_id] = yerbas_cuda_get_device_properties(&cached[device_id], device_id);
    });
    if (result[device_id] != cudaSuccess) return result[device_id];
    *props = cached[device_id];
    return cudaSuccess;
}

// The staggered implementation historically created/destroyed CUDA events on
// every launch. Route those calls through a tiny per-device persistent pool while
// this include is compiled. The legacy scheduler and the rest of the backend keep
// their existing CUDA event behavior unchanged.
#include "cuda/generated/cuda_backend_cn_stagger_event_pool.inc"
#define cn_overlap_selectors_ready cn_stagger_selectors_ready
#ifdef cudaGetDeviceProperties
#undef cudaGetDeviceProperties
#endif
#define cudaGetDeviceProperties cn_stagger_cached_device_properties
#define cudaEventCreateWithFlags cn_stagger_event_create_with_flags
#define cudaEventDestroy cn_stagger_event_release
#include "cuda/generated/cuda_backend_cn_staggered_overlap.inc"
#undef cudaEventDestroy
#undef cudaEventCreateWithFlags
#undef cudaGetDeviceProperties
#define cudaGetDeviceProperties yerbas_cuda_get_device_properties
#undef cn_overlap_selectors_ready
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
#undef cudaGetDeviceProperties
