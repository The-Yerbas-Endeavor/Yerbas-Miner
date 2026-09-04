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

// Keep the runtime selector tiny. Each conventional hash now has a separate
// noinline device entry point so nvcc does not have to inline all 15 algorithms
// into one giant GhostRider stage body. Internal linkage keeps this header safe
// when it is included by multiple CUDA translation units.
static __device__ __noinline__ void core512_blake(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { blake512(input, length, out); }
static __device__ __noinline__ void core512_bmw(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { bmw512(input, length, out); }
static __device__ __noinline__ void core512_groestl(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { groestl512(input, length, out); }
static __device__ __noinline__ void core512_jh(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { jh512(input, length, out); }
static __device__ __noinline__ void core512_keccak(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { keccak512(input, length, out); }
static __device__ __noinline__ void core512_skein(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { skein512(input, length, out); }
static __device__ __noinline__ void core512_luffa(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { luffa512(input, length, out); }
static __device__ __noinline__ void core512_cubehash(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { cubehash512(input, length, out); }
static __device__ __noinline__ void core512_shavite(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { shavite512(input, length, out); }
static __device__ __noinline__ void core512_simd(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { simd512(input, length, out); }
static __device__ __noinline__ void core512_echo(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { echo512(input, length, out); }
static __device__ __noinline__ void core512_hamsi(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { hamsi512(input, length, out); }
static __device__ __noinline__ void core512_fugue(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { fugue512(input, length, out); }
static __device__ __noinline__ void core512_shabal(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { shabal512(input, length, out); }
static __device__ __noinline__ void core512_whirlpool(const std::uint8_t* input, std::size_t length, std::uint8_t out[64]) { whirlpool512(input, length, out); }

// GhostRider core indexes match Yerbas Core HashSelection/coreHash ordering.
// Every true case here is GPU-only; CPU hash fallback is intentionally absent.
__device__ __forceinline__ bool dispatch_core512(std::uint8_t algorithm,
                                                 const std::uint8_t* input,
                                                 std::size_t length,
                                                 std::uint8_t out[64])
{
    switch (algorithm) {
    case 0: core512_blake(input, length, out); return true;
    case 1: core512_bmw(input, length, out); return true;
    case 2: core512_groestl(input, length, out); return true;
    case 3: core512_jh(input, length, out); return true;
    case 4: core512_keccak(input, length, out); return true;
    case 5: core512_skein(input, length, out); return true;
    case 6: core512_luffa(input, length, out); return true;
    case 7: core512_cubehash(input, length, out); return true;
    case 8: core512_shavite(input, length, out); return true;
    case 9: core512_simd(input, length, out); return true;
    case 10: core512_echo(input, length, out); return true;
    case 11: core512_hamsi(input, length, out); return true;
    case 12: core512_fugue(input, length, out); return true;
    case 13: core512_shabal(input, length, out); return true;
    case 14: core512_whirlpool(input, length, out); return true;
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
