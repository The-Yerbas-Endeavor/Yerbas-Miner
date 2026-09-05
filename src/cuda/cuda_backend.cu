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

// Keep the historical 32-hash block-size tuner available for reference, but route
// production through a real-batch geometry selector below. This prevents a tiny
// microbenchmark from overriding a kernel/thread choice measured on thousands of
// hashes.
#define launch_cn_loop_block_tuned launch_cn_loop_block_tuned_micro_legacy
#include "cuda/generated/cuda_backend_cn_blocksize_tuner.inc"
#undef launch_cn_loop_block_tuned

namespace {

constexpr int kCnProductionGeometryPasses = 3;
constexpr int kCnProductionGeometryMaxCandidates = 6;

struct CnProductionGeometryState {
    bool initialized{false};
    bool selected{false};
    std::size_t count{0};
    int mode{0};
    int baseline_threads{0};
    int candidate_count{0};
    int candidate_index{0};
    int sample_index{0};
    std::array<int, kCnProductionGeometryMaxCandidates> threads{};
    std::array<std::array<float, kCnProductionGeometryPasses>,
               kCnProductionGeometryMaxCandidates> times{};
};

static std::array<std::array<CnProductionGeometryState, 6>, kCn2LaneMaxDevices>
    g_cn_production_geometry{};

inline bool cn_geometry_threads_valid(int mode,
                                      int threads,
                                      const cudaDeviceProp& props)
{
    const int warp = std::max(32, props.warpSize);
    if (threads < warp || threads > props.maxThreadsPerBlock ||
        (threads % warp) != 0)
        return false;
    if (mode == 222) {
        int active = 0;
        const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, cryptonight_loop_stage_ttable2_tile64<0>, threads,
            cn_tile64_dynamic_shared_bytes(threads));
        if (rc != cudaSuccess || active <= 0) {
            cudaGetLastError();
            return false;
        }
    }
    return true;
}

template <std::uint8_t VariantIndex>
bool cn_geometry_threads_valid_variant(int mode,
                                        int threads,
                                        const cudaDeviceProp& props)
{
    const int warp = std::max(32, props.warpSize);
    if (threads < warp || threads > props.maxThreadsPerBlock ||
        (threads % warp) != 0)
        return false;
    if (mode == 222) {
        int active = 0;
        const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, cryptonight_loop_stage_ttable2_tile64<VariantIndex>, threads,
            cn_tile64_dynamic_shared_bytes(threads));
        if (rc != cudaSuccess || active <= 0) {
            cudaGetLastError();
            return false;
        }
    }
    return true;
}

template <std::uint8_t VariantIndex>
void initialize_cn_production_geometry(int device_id,
                                       const cudaDeviceProp& props,
                                       std::size_t count,
                                       int mode,
                                       int baseline_threads)
{
    auto& state = g_cn_production_geometry[device_id][VariantIndex];
    if (state.initialized) return;
    state = CnProductionGeometryState{};
    state.initialized = true;
    state.count = count;
    state.mode = mode;
    state.baseline_threads = baseline_threads;

    const auto add = [&](int threads) {
        if (!cn_geometry_threads_valid_variant<VariantIndex>(mode, threads, props)) return;
        for (int i = 0; i < state.candidate_count; ++i)
            if (state.threads[i] == threads) return;
        if (state.candidate_count < kCnProductionGeometryMaxCandidates)
            state.threads[state.candidate_count++] = threads;
    };

    // Architecture-generic production candidates. The selector-provided baseline
    // is always included; unsupported sizes are skipped on smaller devices.
    add(baseline_threads);
    add(32);
    add(128);
    add(256);
    add(512);
    add(768);

    if (state.candidate_count == 0) {
        state.threads[0] = std::max(32, std::min(baseline_threads, props.maxThreadsPerBlock));
        state.candidate_count = 1;
    }

    std::cout << "[CUDA CN production geometry] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | batch=" << count
              << " | mode=" << cn_residency_mode_name(mode)
              << " | baseline=" << baseline_threads
              << " | candidates=";
    for (int i = 0; i < state.candidate_count; ++i) {
        if (i) std::cout << ',';
        std::cout << state.threads[i];
    }
    std::cout << " | passes=" << kCnProductionGeometryPasses
              << " | sampling=real-production-batch\n";
}

template <std::uint8_t VariantIndex>
void record_cn_production_geometry_sample(int device_id,
                                          float elapsed_ms)
{
    auto& state = g_cn_production_geometry[device_id][VariantIndex];
    const int ci = state.candidate_index;
    const int si = state.sample_index;
    state.times[ci][si] = elapsed_ms;

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN geometry sample] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | threads=" << state.threads[ci]
              << " | sample=" << (si + 1) << '/' << kCnProductionGeometryPasses
              << " | elapsed=" << elapsed_ms << " ms"
              << std::defaultfloat << '\n';

    ++state.sample_index;
    if (state.sample_index < kCnProductionGeometryPasses) return;
    state.sample_index = 0;
    ++state.candidate_index;
    if (state.candidate_index < state.candidate_count) return;

    int best_threads = state.baseline_threads;
    float best_ms = 1.0e30F;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN production geometry] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | batch=" << state.count;
    for (int i = 0; i < state.candidate_count; ++i) {
        auto samples = state.times[i];
        const float median = cn_hardened_median(samples);
        std::cout << " | t" << state.threads[i] << '=' << median << " ms";
        if (median < best_ms) {
            best_ms = median;
            best_threads = state.threads[i];
        }
    }
    state.selected = true;
    g_cn_hardened_threads[device_id][VariantIndex] = best_threads;
    std::cout << " | selected=" << best_threads
              << " | selected-ms=" << best_ms
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex, bool UseTTable>
void launch_cn_loop_block_tuned(cudaStream_t stream,
                                std::uint8_t* states,
                                std::size_t count,
                                std::uint8_t* scratchpads,
                                cryptonight::SplitContext* contexts,
                                const CnGeometry& geometry)
{
    (void)states;
    (void)geometry;
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN production geometry failed");
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) {
        cn_launch_mode<VariantIndex>(stream, 4, 128, count, scratchpads, contexts);
        return;
    }

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties CN production geometry failed");

    initialize_cn_production_selector<VariantIndex>(
        stream, device_id, props, count, scratchpads, contexts);
    auto& selector = g_cn_production_selector[device_id][VariantIndex];

    // The existing production kernel selector still owns parity and kernel mode.
    // Let it finish first; its launch path is already measured on the real batch.
    if (!selector.selected) {
        launch_cn_loop_block_tuned_micro_legacy<VariantIndex, UseTTable>(
            stream, states, count, scratchpads, contexts, geometry);
        return;
    }

    int mode = g_cn_hardened_mode[device_id][VariantIndex];
    int baseline_threads = g_cn_hardened_threads[device_id][VariantIndex];
    if (mode == 0) mode = 4;
    if (baseline_threads <= 0) baseline_threads = 128;

    auto& tune = g_cn_production_geometry[device_id][VariantIndex];
    if (!tune.initialized)
        initialize_cn_production_geometry<VariantIndex>(
            device_id, props, count, mode, baseline_threads);

    // Stagger partitions must not reset or contaminate the full-batch geometry
    // trial. Until tuning completes they use the selector's safe baseline.
    if (!tune.selected && tune.count != count) {
        cn_launch_mode<VariantIndex>(stream, mode, tune.baseline_threads,
                                     count, scratchpads, contexts);
        return;
    }

    if (!tune.selected) {
        const int threads = tune.threads[tune.candidate_index];
        cudaEvent_t start{}, stop{};
        check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault),
                   "cudaEventCreate CN geometry start failed");
        check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault),
                   "cudaEventCreate CN geometry stop failed");
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN geometry start failed");
        cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN geometry stop failed");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN geometry failed");
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "cudaEventElapsedTime CN geometry failed");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        record_cn_production_geometry_sample<VariantIndex>(device_id, elapsed_ms);
        return;
    }

    cn_launch_mode<VariantIndex>(stream, mode,
                                 g_cn_hardened_threads[device_id][VariantIndex],
                                 count, scratchpads, contexts);
}

} // anonymous namespace

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

// Stagger may begin only after both the production kernel selector and the new
// full-batch geometry selector have finished. That keeps partition launches from
// starving or contaminating either trial sequence.
template <std::uint8_t VariantIndex>
bool cn_stagger_selectors_ready(int device_id, bool)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    if (device_id < 0 || device_id >= kCnPhaseProfileMaxDevices) return false;
    return g_cn_production_selector[device_id][VariantIndex].selected &&
           g_cn_production_geometry[device_id][VariantIndex].selected;
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
