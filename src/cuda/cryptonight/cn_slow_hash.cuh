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

__device__ __forceinline__ std::uint64_t cn_load64(const std::uint8_t* p)
{
    return load64le(p);
}

__device__ __forceinline__ void cn_store64(std::uint8_t* p, std::uint64_t v)
{
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (i * 8));
}

__device__ __forceinline__ void copy16(std::uint8_t* dst, const std::uint8_t* src)
{
    #pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = src[i];
}

__device__ __forceinline__ void xor16(std::uint8_t* dst, const std::uint8_t* src)
{
    #pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] ^= src[i];
}

__device__ __forceinline__ std::size_t cn_index(const std::uint8_t block[16], std::size_t aes_rounds)
{
    return static_cast<std::size_t>((cn_load64(block) >> 4) & (aes_rounds - 1));
}

__device__ __forceinline__ void variant1_mutate(std::uint8_t block[16])
{
    const std::uint8_t tmp = block[11];
    constexpr std::uint32_t table = 0x75310U;
    const std::uint8_t index = static_cast<std::uint8_t>((((tmp >> 3) & 6U) | (tmp & 1U)) << 1);
    block[11] = static_cast<std::uint8_t>(tmp ^ ((table >> index) & 0x30U));
}

__device__ __forceinline__ bool slow_hash(std::uint8_t variant_index,
                                          const std::uint8_t* input,
                                          std::size_t length,
                                          std::uint8_t* scratchpad,
                                          std::uint8_t out[32])
{
    if (variant_index >= 6 || scratchpad == nullptr || input == nullptr || length < 43)
        return false;

    const VariantConfig cfg = config_value(variant_index);

    std::uint8_t state[200];
    keccak1600(input, length, state);

    std::uint8_t text[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) text[i] = state[64 + i];

    std::uint8_t expanded[240];
    aes256_expand_key(state, expanded);

    const std::size_t init_rounds = cfg.page_size / 128U;
    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block)
            aes_pseudo_round(text + block * 16, expanded);
        for (int b = 0; b < 128; ++b) scratchpad[i * 128U + static_cast<std::size_t>(b)] = text[b];
    }

    // Yerbas Core keeps two 16-byte b blocks. The upper half preserves the
    // previous b value before the lower half is replaced with c at the end of
    // every CryptoNight iteration. Keep that state exactly instead of
    // collapsing b to a single block.
    std::uint8_t a[16], b[32], c[16], t[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<std::uint8_t>(state[i] ^ state[32 + i]);
        b[i] = static_cast<std::uint8_t>(state[16 + i] ^ state[48 + i]);
        b[16 + i] = 0;
    }

    const std::uint64_t tweak = cn_load64(input + 35) ^ cn_load64(state + 192);

    for (std::uint32_t i = 0; i < cfg.iterations; ++i) {
        std::size_t j = cn_index(a, cfg.aes_rounds);
        std::uint8_t* slot = scratchpad + j * 16U;

        aes_single_round(slot, c, a);
        #pragma unroll
        for (int k = 0; k < 16; ++k) slot[k] = static_cast<std::uint8_t>(c[k] ^ b[k]);
        variant1_mutate(slot);

        j = cn_index(c, cfg.aes_rounds);
        slot = scratchpad + j * 16U;
        copy16(t, slot);

        const std::uint64_t c0 = cn_load64(c);
        const std::uint64_t t0 = cn_load64(t);
        const std::uint64_t lo = c0 * t0;
        const std::uint64_t hi = __umul64hi(c0, t0);

        std::uint64_t a0 = cn_load64(a) + hi;
        std::uint64_t a1 = cn_load64(a + 8) + lo;
        cn_store64(slot, a0);
        cn_store64(slot + 8, a1 ^ tweak);

        a0 ^= t0;
        a1 ^= cn_load64(t + 8);
        cn_store64(a, a0);
        cn_store64(a + 8, a1);

        // Core: copy_block(b + AES_BLOCK_SIZE, b); copy_block(b, c);
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

} // namespace yerbas::cuda::cryptonight
