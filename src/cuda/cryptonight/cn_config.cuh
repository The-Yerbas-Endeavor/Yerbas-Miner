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

    // Memory and SM concurrency are already hard runtime-derived limits. Do not
    // add a fixed hash-count ceiling here: a static cap silently penalizes GPUs
    // with substantially more memory or SM capacity than the hardware available
    // when the miner was first tuned. The live BatchEngine still enforces exact
    // allocation bounds before launch.
    std::size_t selected = memory_limit < sm_limit ? memory_limit : sm_limit;
    selected = (selected / kAlignment) * kAlignment;
    if (selected < kAlignment) selected = kAlignment;
    return selected;
}

} // namespace detail

// Throughput-first production sizing. Keep a device-independent reserve for the
// display/driver/runtime, but allow substantially more of the currently-free
// VRAM and SM concurrency to participate in the heavy 2MiB CryptoNight class.
// The live BatchEngine still applies its exact allocated-byte guard before any
// job batch is launched, so this only widens the safe search space.
inline std::size_t adaptive_batch_limit(int device_id)
{
    return detail::adaptive_batch_from_budget(device_id, 82U, 192U, 1024U);
}

// Deep-tune capacity is intentionally wider than the initial production batch.
// It is derived only from runtime memory/SM limits (never GPU names or compute
// generations) and leaves an 8% free-VRAM reserve. Calibration/parity gates then
// decide whether any of this headroom is actually worth using.
inline std::size_t adaptive_batch_capacity(int device_id)
{
    const std::size_t production = adaptive_batch_limit(device_id);
    const std::size_t capacity = detail::adaptive_batch_from_budget(device_id, 92U, 256U, production);
    return capacity < production ? production : capacity;
}

#define kInitialMaxBatch adaptive_batch_limit(id)
#define kInitialBatchCapacity adaptive_batch_capacity(id)

} // namespace yerbas::cuda::cryptonight
