#pragma once

#include <array>
#include <cstdint>

namespace yerbas::cuda {

struct CoreCoverage {
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

inline constexpr unsigned int implemented_core_count()
{
    unsigned int count = 0;
    for (const auto& core : kCoreCoverage) {
        if (core.implemented) ++count;
    }
    return count;
}

} // namespace yerbas::cuda
