#pragma once

#include "cuda/cryptonight/cn_slow_hash.cuh"

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::cryptonight {

// Persistent state carried between the three split CryptoNight kernels.
// Keeping the 200-byte Keccak state and loop seeds in global memory lets the
// scratchpad-fill, memory-hard loop, and collapse/finalizer kernels have much
// smaller independent live ranges than the original monolithic slow_hash().
struct alignas(16) SplitContext {
    std::uint8_t state[200];
    std::uint8_t a[16];
    std::uint8_t b[32];
    std::uint64_t tweak;
};

template <std::uint8_t VariantIndex>
__device__ __forceinline__ bool split_setup(const std::uint8_t* input,
                                            std::size_t length,
                                            std::uint8_t* scratchpad,
                                            SplitContext& ctx)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    if (scratchpad == nullptr || input == nullptr || length < 43) return false;

    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;

    keccak1600(input, length, ctx.state);

    alignas(16) std::uint8_t text[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) text[i] = ctx.state[64 + i];

    alignas(16) std::uint8_t expanded[240];
    aes256_expand_key(ctx.state, expanded);

    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block)
            aes_pseudo_round(text + block * 16, expanded);
        #pragma unroll 16
        for (int b = 0; b < 128; ++b)
            scratchpad[i * 128U + static_cast<std::size_t>(b)] = text[b];
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        ctx.a[i] = static_cast<std::uint8_t>(ctx.state[i] ^ ctx.state[32 + i]);
        ctx.b[i] = static_cast<std::uint8_t>(ctx.state[16 + i] ^ ctx.state[48 + i]);
        ctx.b[16 + i] = 0;
    }

    ctx.tweak = cn_load64(input + 35) ^ cn_load64_aligned(ctx.state + 192);
    return true;
}

template <std::uint8_t VariantIndex>
__device__ __forceinline__ void split_memory_loop(std::uint8_t* scratchpad,
                                                  const SplitContext& ctx)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;

    alignas(16) std::uint8_t a[16];
    alignas(16) std::uint8_t b[32];
    alignas(16) std::uint8_t c[16];
    alignas(16) std::uint8_t t[16];
    copy16(a, ctx.a);
    copy16(b, ctx.b);
    copy16(b + 16, ctx.b + 16);
    const std::uint64_t tweak = ctx.tweak;

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
}

template <std::uint8_t VariantIndex>
__device__ __forceinline__ void split_finalize(std::uint8_t* scratchpad,
                                               SplitContext& ctx,
                                               std::uint8_t out[32])
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;

    alignas(16) std::uint8_t text[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) text[i] = ctx.state[64 + i];

    alignas(16) std::uint8_t expanded[240];
    aes256_expand_key(ctx.state + 32, expanded);
    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block) {
            std::uint8_t* x = text + block * 16;
            xor16(x, scratchpad + i * 128U + static_cast<std::size_t>(block * 16));
            aes_pseudo_round(x, expanded);
        }
    }

    #pragma unroll
    for (int i = 0; i < 128; ++i) ctx.state[64 + i] = text[i];
    keccak_permute_state(ctx.state);
    dispatch_extra_hash(static_cast<std::uint8_t>(ctx.state[0] & 3U), ctx.state, out);
}

} // namespace yerbas::cuda::cryptonight
