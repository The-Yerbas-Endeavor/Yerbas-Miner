#pragma once

#include "cuda/core/blake512.cuh"
#include "cuda/core/bmw512.cuh"
#include "cuda/core/cubehash512.cuh"
#include "cuda/core/echo512.cuh"
#include "cuda/core/fugue512.cuh"
#include "cuda/core/groestl512.cuh"
#include "cuda/core/hamsi512.cuh"
#include "cuda/core/jh512.cuh"
#include "cuda/core/keccak512.cuh"
#include "cuda/core/luffa512.cuh"
#include "cuda/core/shabal512.cuh"
#include "cuda/core/shavite512.cuh"
#include "cuda/core/simd512.cuh"
#include "cuda/core/skein512.cuh"
#include "cuda/core/whirlpool512.cuh"

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::core {

// Keep each conventional GhostRider hash in its own device function. The
// production stage algorithm is uniform across the entire launch, so inlining
// all 15 implementations into one giant kernel only increases instruction-cache
// and register/local-memory pressure. The tiny runtime switch remains inline,
// while the selected algorithm executes through one compile-time specialization.
// This preserves exact hash behavior and remains architecture/model agnostic.
template <std::uint8_t Algorithm>
__device__ __noinline__ void dispatch_core512_fixed(const std::uint8_t* input,
                                                     std::size_t length,
                                                     std::uint8_t out[64])
{
    static_assert(Algorithm <= 14, "invalid GhostRider conventional hash index");
    if constexpr (Algorithm == 0) blake512(input, length, out);
    else if constexpr (Algorithm == 1) bmw512(input, length, out);
    else if constexpr (Algorithm == 2) groestl512(input, length, out);
    else if constexpr (Algorithm == 3) jh512(input, length, out);
    else if constexpr (Algorithm == 4) keccak512(input, length, out);
    else if constexpr (Algorithm == 5) skein512(input, length, out);
    else if constexpr (Algorithm == 6) luffa512(input, length, out);
    else if constexpr (Algorithm == 7) cubehash512(input, length, out);
    else if constexpr (Algorithm == 8) shavite512(input, length, out);
    else if constexpr (Algorithm == 9) simd512(input, length, out);
    else if constexpr (Algorithm == 10) echo512(input, length, out);
    else if constexpr (Algorithm == 11) hamsi512(input, length, out);
    else if constexpr (Algorithm == 12) fugue512(input, length, out);
    else if constexpr (Algorithm == 13) shabal512(input, length, out);
    else if constexpr (Algorithm == 14) whirlpool512(input, length, out);
}

// GhostRider core indexes match Yerbas Core HashSelection/coreHash ordering.
// Every true case here is GPU-only; CPU hash fallback is intentionally absent.
__device__ __forceinline__ bool dispatch_core512(std::uint8_t algorithm,
                                                 const std::uint8_t* input,
                                                 std::size_t length,
                                                 std::uint8_t out[64])
{
    switch (algorithm) {
    case 0: dispatch_core512_fixed<0>(input, length, out); return true;
    case 1: dispatch_core512_fixed<1>(input, length, out); return true;
    case 2: dispatch_core512_fixed<2>(input, length, out); return true;
    case 3: dispatch_core512_fixed<3>(input, length, out); return true;
    case 4: dispatch_core512_fixed<4>(input, length, out); return true;
    case 5: dispatch_core512_fixed<5>(input, length, out); return true;
    case 6: dispatch_core512_fixed<6>(input, length, out); return true;
    case 7: dispatch_core512_fixed<7>(input, length, out); return true;
    case 8: dispatch_core512_fixed<8>(input, length, out); return true;
    case 9: dispatch_core512_fixed<9>(input, length, out); return true;
    case 10: dispatch_core512_fixed<10>(input, length, out); return true;
    case 11: dispatch_core512_fixed<11>(input, length, out); return true;
    case 12: dispatch_core512_fixed<12>(input, length, out); return true;
    case 13: dispatch_core512_fixed<13>(input, length, out); return true;
    case 14: dispatch_core512_fixed<14>(input, length, out); return true;
    default: return false;
    }
}

__host__ __device__ constexpr bool core512_implemented(std::uint8_t algorithm)
{
    return algorithm <= 14;
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
