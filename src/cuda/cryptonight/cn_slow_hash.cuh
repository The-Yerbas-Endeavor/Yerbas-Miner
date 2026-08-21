#pragma once

#include "cuda/cryptonight/cn_aes.cuh"
#include "cuda/cryptonight/cn_config.cuh"
#include "cuda/cryptonight/cn_keccak.cuh"

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::cryptonight {

// Implemented by cn_final.cuh from the pinned Yerbas Core c_* extra hashes.
__device__ __forceinline__ void dispatch_extra_hash(std::uint8_t selector,
                                                    const std::uint8_t state[200],
                                                    std::uint8_t out[32]);

// Generic little-endian load retained for the intentionally unaligned
// variant-1 input tweak at input+35 and validator scratch state.
__device__ __forceinline__ std::uint64_t cn_load64(const std::uint8_t* p)
{
    return load64le(p);
}

// All production hot-loop state blocks and scratchpad slots are explicitly
// 16-byte aligned. Use native 64-bit accesses there instead of eight byte
// loads, shifts and stores per word.
__device__ __forceinline__ std::uint64_t cn_load64_aligned(const std::uint8_t* p)
{
    return *reinterpret_cast<const std::uint64_t*>(p);
}

__device__ __forceinline__ void cn_store64_aligned(std::uint8_t* p, std::uint64_t v)
{
    *reinterpret_cast<std::uint64_t*>(p) = v;
}

__device__ __forceinline__ void copy16(std::uint8_t* dst, const std::uint8_t* src)
{
    reinterpret_cast<std::uint64_t*>(dst)[0] = reinterpret_cast<const std::uint64_t*>(src)[0];
    reinterpret_cast<std::uint64_t*>(dst)[1] = reinterpret_cast<const std::uint64_t*>(src)[1];
}

__device__ __forceinline__ void xor16(std::uint8_t* dst, const std::uint8_t* src)
{
    reinterpret_cast<std::uint64_t*>(dst)[0] ^= reinterpret_cast<const std::uint64_t*>(src)[0];
    reinterpret_cast<std::uint64_t*>(dst)[1] ^= reinterpret_cast<const std::uint64_t*>(src)[1];
}

__device__ __forceinline__ std::size_t cn_index_masked(const std::uint8_t block[16], std::size_t mask)
{
    return static_cast<std::size_t>((cn_load64_aligned(block) >> 4) & mask);
}

// Generic validator/reference helper. Validation scratch arrays are not
// guaranteed to have the production path's explicit 16-byte alignment, so
// keep this helper on the safe byte-oriented load path.
__device__ __forceinline__ std::size_t cn_index(const std::uint8_t block[16], std::size_t aes_rounds)
{
    return static_cast<std::size_t>((cn_load64(block) >> 4) & (aes_rounds - 1U));
}

__device__ __forceinline__ void variant1_mutate(std::uint8_t block[16])
{
    const std::uint8_t tmp = block[11];
    constexpr std::uint32_t table = 0x75310U;
    const std::uint8_t index = static_cast<std::uint8_t>((((tmp >> 3) & 6U) | (tmp & 1U)) << 1);
    block[11] = static_cast<std::uint8_t>(tmp ^ ((table >> index) & 0x30U));
}

// GhostRider's six CryptoNight flavors are fixed per stage. Specializing the
// full slow-hash path at compile time lets NVCC constant-fold page sizing,
// address masks and iteration counts instead of carrying a runtime config
// through the hottest memory-dependent loop.
template <std::uint8_t VariantIndex>
__device__ __forceinline__ bool slow_hash_specialized(const std::uint8_t* input,
                                                       std::size_t length,
                                                       std::uint8_t* scratchpad,
                                                       std::uint8_t out[32])
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    if (scratchpad == nullptr || input == nullptr || length < 43) return false;

    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    constexpr std::size_t init_rounds = cfg.page_size / 128U;

    alignas(16) std::uint8_t state[200];
    keccak1600(input, length, state);

    alignas(16) std::uint8_t text[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) text[i] = state[64 + i];

    alignas(16) std::uint8_t expanded[240];
    aes256_expand_key(state, expanded);

    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block)
            aes_pseudo_round(text + block * 16, expanded);
        #pragma unroll 16
        for (int b = 0; b < 128; ++b)
            scratchpad[i * 128U + static_cast<std::size_t>(b)] = text[b];
    }

    alignas(16) std::uint8_t a[16];
    alignas(16) std::uint8_t b[32];
    alignas(16) std::uint8_t c[16];
    alignas(16) std::uint8_t t[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<std::uint8_t>(state[i] ^ state[32 + i]);
        b[i] = static_cast<std::uint8_t>(state[16 + i] ^ state[48 + i]);
        b[16 + i] = 0;
    }

    const std::uint64_t tweak = cn_load64(input + 35) ^ cn_load64_aligned(state + 192);

    #pragma unroll 1
    for (std::uint32_t i = 0; i < cfg.iterations; ++i) {
        std::size_t j = cn_index_masked(a, address_mask);
        std::uint8_t* slot = scratchpad + j * 16U;

        aes_single_round(slot, c, a);
        reinterpret_cast<std::uint64_t*>(slot)[0] =
            reinterpret_cast<const std::uint64_t*>(c)[0] ^ reinterpret_cast<const std::uint64_t*>(b)[0];
        reinterpret_cast<std::uint64_t*>(slot)[1] =
            reinterpret_cast<const std::uint64_t*>(c)[1] ^ reinterpret_cast<const std::uint64_t*>(b)[1];
        variant1_mutate(slot);

        j = cn_index_masked(c, address_mask);
        slot = scratchpad + j * 16U;
        copy16(t, slot);

        const std::uint64_t c0 = cn_load64_aligned(c);
        const std::uint64_t t0 = cn_load64_aligned(t);
        const std::uint64_t lo = c0 * t0;
        const std::uint64_t hi = __umul64hi(c0, t0);

        std::uint64_t a0 = cn_load64_aligned(a) + hi;
        std::uint64_t a1 = cn_load64_aligned(a + 8) + lo;
        cn_store64_aligned(slot, a0);
        cn_store64_aligned(slot + 8, a1 ^ tweak);

        a0 ^= t0;
        a1 ^= cn_load64_aligned(t + 8);
        cn_store64_aligned(a, a0);
        cn_store64_aligned(a + 8, a1);

        copy16(b + 16, b);
        copy16(b, c);
    }

    #pragma unroll
    for (int i = 0; i < 128; ++i) text[i] = state[64 + i];
    aes256_expand_key(state + 32, expanded);
    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block) {
            std::uint8_t* x = text + block * 16;
            xor16(x, scratchpad + i * 128U + static_cast<std::size_t>(block * 16));
            aes_pseudo_round(x, expanded);
        }
    }

    #pragma unroll
    for (int i = 0; i < 128; ++i) state[64 + i] = text[i];
    keccak_permute_state(state);
    dispatch_extra_hash(static_cast<std::uint8_t>(state[0] & 3U), state, out);
    return true;
}

__device__ __forceinline__ bool slow_hash(std::uint8_t variant_index,
                                          const std::uint8_t* input,
                                          std::size_t length,
                                          std::uint8_t* scratchpad,
                                          std::uint8_t out[32])
{
    switch (variant_index) {
    case 0: return slow_hash_specialized<0>(input, length, scratchpad, out);
    case 1: return slow_hash_specialized<1>(input, length, scratchpad, out);
    case 2: return slow_hash_specialized<2>(input, length, scratchpad, out);
    case 3: return slow_hash_specialized<3>(input, length, scratchpad, out);
    case 4: return slow_hash_specialized<4>(input, length, scratchpad, out);
    case 5: return slow_hash_specialized<5>(input, length, scratchpad, out);
    default: return false;
    }
}

} // namespace yerbas::cuda::cryptonight
