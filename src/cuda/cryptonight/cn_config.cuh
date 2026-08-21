#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

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

// Constant-memory AES moved the Pascal CryptoNight path from ~50 H/s to
// ~228 H/s and throughput was still increasing at the old 1536-hash ceiling.
// A GTX 1080 Ti has ~11 GiB; CN-Fast needs 2 MiB/hash, so 4096 hashes consume
// ~8 GiB of scratchpad and leave practical headroom for states and CUDA
// runtime allocations. Benchmark up to this ceiling before changing the
// kernel architecture.
inline constexpr std::size_t kInitialMaxBatch = 4096;

} // namespace yerbas::cuda::cryptonight
