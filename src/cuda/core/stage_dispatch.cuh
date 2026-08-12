#pragma once

#include "cuda/core/blake512.cuh"
#include "cuda/core/cubehash512.cuh"
#include "cuda/core/keccak512.cuh"
#include "cuda/core/skein512.cuh"

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::core {

// GhostRider core indexes match Yerbas Core HashSelection/coreHash ordering.
// This dispatcher intentionally returns false for algorithms that have not yet
// been implemented and validated on CUDA.  Callers must never treat false as a
// valid digest.
__device__ __forceinline__ bool dispatch_core512(std::uint8_t algorithm,
                                                 const std::uint8_t* input,
                                                 std::size_t length,
                                                 std::uint8_t out[64])
{
    switch (algorithm) {
    case 0: // BLAKE-512
        blake512(input, length, out);
        return true;
    case 4: // Keccak-512
        keccak512(input, length, out);
        return true;
    case 5: // Skein-512
        skein512(input, length, out);
        return true;
    case 7: // CubeHash-512
        cubehash512(input, length, out);
        return true;
    default:
        return false;
    }
}

__host__ __device__ constexpr bool core512_implemented(std::uint8_t algorithm)
{
    return algorithm == 0 || algorithm == 4 || algorithm == 5 || algorithm == 7;
}

__host__ __device__ constexpr const char* core512_name(std::uint8_t algorithm)
{
    switch (algorithm) {
    case 0: return "BLAKE-512";
    case 1: return "BMW-512";
    case 2: return "Groestl-512";
    case 3: return "JH-512";
    case 4: return "Keccak-512";
    case 5: return "Skein-512";
    case 6: return "Luffa-512";
    case 7: return "CubeHash-512";
    case 8: return "Shavite-512";
    case 9: return "SIMD-512";
    case 10: return "Echo-512";
    case 11: return "Hamsi-512";
    case 12: return "Fugue-512";
    case 13: return "Shabal-512";
    case 14: return "Whirlpool";
    default: return "unknown";
    }
}

} // namespace yerbas::cuda::core
