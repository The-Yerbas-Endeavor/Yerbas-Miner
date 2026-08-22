#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace yerbas::cuda::cryptonight {

// Exact parameters from the pinned Yerbas Core slow-hash.h wrappers.
// All six GhostRider variants use CryptoNight variant 1 semantics.
struct VariantConfig {
    std::uint8_t index;
    const char* name;
    std::uint32_t page_size;
    std::uint32_t iterations;
    std::size_t aes_rounds;
    int variant;
};

inline constexpr std::array<VariantConfig, 6> kVariantConfigs{{
    {0, "CN-Dark",       524288U,  131072U,  32768U,  1},
    {1, "CN-DarkLite",   524288U,  131072U,  16384U,  1},
    {2, "CN-Fast",      2097152U,  262144U, 131072U,  1},
    {3, "CN-Lite",      1048576U,  262144U,  65536U,  1},
    {4, "CN-Turtle",     262144U,   65536U,  16384U,  1},
    {5, "CN-TurtleLite", 262144U,   65536U,   8192U,  1},
}};

// Keep the host constexpr table above for host-side sizing and validation,
// but return literal device-local configs when called from CUDA code. This
// avoids taking a device pointer into std::array host storage.
__host__ __device__ inline constexpr VariantConfig config_value(std::uint8_t index)
{
    switch (index) {
    case 0: return {0, "CN-Dark",       524288U,  131072U,  32768U,  1};
    case 1: return {1, "CN-DarkLite",   524288U,  131072U,  16384U,  1};
    case 2: return {2, "CN-Fast",      2097152U,  262144U, 131072U,  1};
    case 3: return {3, "CN-Lite",      1048576U,  262144U,  65536U,  1};
    case 4: return {4, "CN-Turtle",     262144U,   65536U,  16384U,  1};
    case 5: return {5, "CN-TurtleLite", 262144U,   65536U,   8192U,  1};
    default: return {255, nullptr, 0U, 0U, 0U, 0};
    }
}

inline constexpr const VariantConfig* config(std::uint8_t index)
{
    return index < kVariantConfigs.size() ? &kVariantConfigs[index] : nullptr;
}

inline constexpr std::size_t max_scratchpad_bytes()
{
    std::size_t value = 0;
    for (const auto& item : kVariantConfigs)
        if (item.page_size > value) value = item.page_size;
    return value;
}

// Hardware-adaptive batch ceiling used by BatchEngine auto mode.
//
// GhostRider's largest CryptoNight variant consumes 2 MiB of scratchpad per
// in-flight hash, so blindly using one fixed batch for every GPU either leaves
// large cards under-filled or pushes smaller cards into memory pressure/OOM.
// The auto ceiling combines two independent limits:
//   1. roughly 70% of currently free VRAM, leaving headroom for the driver,
//      desktop, CUDA contexts, states, and split-CN bookkeeping;
//   2. 128 in-flight hashes per SM, which scales the working set with the
//      amount of execution hardware instead of a specific GPU model.
//
// The result is aligned to 128 hashes and capped at 16384 for a conservative
// first-generation autotuner. Explicit smaller user batches remain untouched
// because BatchEngine still takes min(requested_size, this ceiling).
inline std::size_t adaptive_batch_limit(int device_id)
{
    constexpr std::size_t kFallbackBatch = 1024;
    constexpr std::size_t kAlignment = 128;
    constexpr std::size_t kAbsoluteCap = 16384;
    constexpr std::size_t kPerHashOverhead = 4096;

    int previous_device = 0;
    const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
    if (cudaSetDevice(device_id) != cudaSuccess) return kFallbackBatch;

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaDeviceProp props{};
    const bool memory_ok = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
    const bool props_ok = cudaGetDeviceProperties(&props, device_id) == cudaSuccess;

    if (have_previous && previous_device != device_id) cudaSetDevice(previous_device);

    if (!memory_ok || !props_ok || props.multiProcessorCount <= 0) return kFallbackBatch;

    const std::size_t bytes_per_hash = max_scratchpad_bytes() + kPerHashOverhead;
    const std::size_t memory_budget = (free_bytes * 70U) / 100U;
    std::size_t memory_limit = memory_budget / bytes_per_hash;
    std::size_t sm_limit = static_cast<std::size_t>(props.multiProcessorCount) * 128U;

    std::size_t selected = memory_limit < sm_limit ? memory_limit : sm_limit;
    if (selected > kAbsoluteCap) selected = kAbsoluteCap;
    selected = (selected / kAlignment) * kAlignment;
    if (selected < kAlignment) selected = kAlignment;
    return selected;
}

// Compatibility token consumed by the existing BatchEngine constructor.
// The constructor has its device id in scope as `id`, so this expands the old
// fixed ceiling into the per-device adaptive ceiling without changing public
// BatchEngine APIs. Keep this local to CUDA compilation until the next backend
// API cleanup moves auto-batch selection into an explicit constructor helper.
#define kInitialMaxBatch adaptive_batch_limit(id)

} // namespace yerbas::cuda::cryptonight
