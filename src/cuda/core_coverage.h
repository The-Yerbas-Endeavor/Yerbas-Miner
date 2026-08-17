#pragma once

#include <array>
#include <cstdint>

namespace yerbas::cuda {

struct CoreCoverage {
    std::uint8_t index;
    const char* name;
    bool implemented;
};

struct CryptoNightCoverage {
    std::uint8_t index;
    const char* name;
    bool implemented;
};

inline constexpr std::array<CoreCoverage, 15> kCoreCoverage{{
    {0,  "BLAKE-512",    true},
    {1,  "BMW-512",      false},
    {2,  "Groestl-512",  false},
    {3,  "JH-512",       false},
    {4,  "Keccak-512",   true},
    {5,  "Skein-512",    true},
    {6,  "Luffa-512",    false},
    {7,  "CubeHash-512", true},
    {8,  "Shavite-512",  false},
    {9,  "SIMD-512",     false},
    {10, "Echo-512",     false},
    {11, "Hamsi-512",    false},
    {12, "Fugue-512",    false},
    {13, "Shabal-512",   false},
    {14, "Whirlpool",    false},
}};

// Yerbas Core's GhostRider selector can choose one of six CryptoNight
// variants for each of the three memory-hard positions in the 18-stage chain.
// Keep these false until a device implementation is validated byte-for-byte
// against the pinned Yerbas Core reference.
inline constexpr std::array<CryptoNightCoverage, 6> kCryptoNightCoverage{{
    {0, "CN-Dark",       false},
    {1, "CN-DarkLite",   false},
    {2, "CN-Fast",       false},
    {3, "CN-Lite",       false},
    {4, "CN-Turtle",     false},
    {5, "CN-TurtleLite", false},
}};

inline constexpr unsigned int implemented_core_count()
{
    unsigned int count = 0;
    for (const auto& core : kCoreCoverage) {
        if (core.implemented) ++count;
    }
    return count;
}

inline constexpr unsigned int implemented_cryptonight_count()
{
    unsigned int count = 0;
    for (const auto& variant : kCryptoNightCoverage) {
        if (variant.implemented) ++count;
    }
    return count;
}

inline constexpr bool conventional_cores_ready()
{
    return implemented_core_count() == kCoreCoverage.size();
}

inline constexpr bool cryptonight_variants_ready()
{
    return implemented_cryptonight_count() == kCryptoNightCoverage.size();
}

inline constexpr bool full_ghostrider_cuda_coverage()
{
    return conventional_cores_ready() && cryptonight_variants_ready();
}

} // namespace yerbas::cuda
