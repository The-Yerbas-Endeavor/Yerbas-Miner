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

// A full mining batch cannot use the conventional-core 65536 nonce batch:
// CN-Fast requires 2 MiB of scratchpad per active hash. Start conservatively
// with at most 256 concurrent hashes (512 MiB scratchpad per GPU) and tune
// upward later through the intensity policy once correctness is established.
inline constexpr std::size_t kInitialMaxBatch = 256;

} // namespace yerbas::cuda::cryptonight
