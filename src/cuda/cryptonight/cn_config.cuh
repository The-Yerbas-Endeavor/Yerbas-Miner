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

namespace detail {

inline std::size_t adaptive_batch_from_budget(int device_id,
                                              unsigned int memory_percent,
                                              std::size_t hashes_per_sm,
                                              std::size_t fallback)
{
    constexpr std::size_t kAlignment = 128;
    constexpr std::size_t kAbsoluteCap = 16384;
    constexpr std::size_t kPerHashOverhead = 4096;

    int previous_device = 0;
    const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
    if (cudaSetDevice(device_id) != cudaSuccess) return fallback;

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaDeviceProp props{};
    const bool memory_ok = cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
    const bool props_ok = cudaGetDeviceProperties(&props, device_id) == cudaSuccess;

    if (have_previous && previous_device != device_id) cudaSetDevice(previous_device);
    if (!memory_ok || !props_ok || props.multiProcessorCount <= 0) return fallback;

    const std::size_t bytes_per_hash = max_scratchpad_bytes() + kPerHashOverhead;
    const std::size_t memory_budget = (free_bytes * memory_percent) / 100U;
    const std::size_t memory_limit = memory_budget / bytes_per_hash;
    const std::size_t sm_limit = static_cast<std::size_t>(props.multiProcessorCount) * hashes_per_sm;

    std::size_t selected = memory_limit < sm_limit ? memory_limit : sm_limit;
    if (selected > kAbsoluteCap) selected = kAbsoluteCap;
    selected = (selected / kAlignment) * kAlignment;
    if (selected < kAlignment) selected = kAlignment;
    return selected;
}

} // namespace detail

// Production auto batch remains deliberately conservative. This value is also
// the batch size used by the existing CryptoNight tuning/cache keys.
inline std::size_t adaptive_batch_limit(int device_id)
{
    return detail::adaptive_batch_from_budget(device_id, 70U, 128U, 1024U);
}

// Allocate modest headroom above the production batch so active-batch throughput
// experiments can move upward without reallocating buffers or invalidating the
// already-proven CryptoNight tuning. The extra capacity still caps scratchpad
// use at roughly 80% of currently free VRAM and 160 hashes/SM.
inline std::size_t adaptive_batch_capacity(int device_id)
{
    const std::size_t production = adaptive_batch_limit(device_id);
    const std::size_t capacity = detail::adaptive_batch_from_budget(device_id, 80U, 160U, production);
    return capacity < production ? production : capacity;
}

#define kInitialMaxBatch adaptive_batch_limit(id)
#define kInitialBatchCapacity adaptive_batch_capacity(id)

} // namespace yerbas::cuda::cryptonight
