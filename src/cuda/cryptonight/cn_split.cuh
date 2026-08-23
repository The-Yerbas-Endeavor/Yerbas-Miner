#pragma once

#include "cuda/cryptonight/cn_slow_hash.cuh"

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::cryptonight {

struct alignas(16) SplitContext {
    std::uint8_t state[200];
    std::uint8_t a[16];
    std::uint8_t b[16];
    std::uint64_t tweak;
};

template <std::uint8_t VariantIndex, bool UseTTable = false>
__device__ __forceinline__ bool split_setup(const std::uint8_t* input,
                                            std::size_t length,
                                            std::uint8_t* scratchpad,
                                            SplitContext& ctx,
                                            const std::uint32_t* aes_tables = nullptr)
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
        for (int block = 0; block < 8; ++block) {
            if constexpr (UseTTable)
                aes_pseudo_round_ttable(text + block * 16, expanded, aes_tables);
            else
                aes_pseudo_round(text + block * 16, expanded);
        }
        #pragma unroll 16
        for (int b = 0; b < 128; ++b)
            scratchpad[i * 128U + static_cast<std::size_t>(b)] = text[b];
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        ctx.a[i] = static_cast<std::uint8_t>(ctx.state[i] ^ ctx.state[32 + i]);
        ctx.b[i] = static_cast<std::uint8_t>(ctx.state[16 + i] ^ ctx.state[48 + i]);
    }

    ctx.tweak = cn_load64(input + 35) ^ cn_load64_aligned(ctx.state + 192);
    return true;
}

template <std::size_t AddressMask, bool UseTTable = false>
__device__ __forceinline__ void split_memory_iteration(std::uint8_t* scratchpad,
                                                        std::uint8_t a[16],
                                                        std::uint8_t b[16],
                                                        std::uint8_t c[16],
                                                        std::uint8_t t[16],
                                                        std::uint64_t tweak,
                                                        const std::uint32_t* aes_tables = nullptr)
{
    std::size_t j = cn_index_masked(a, AddressMask);
    std::uint8_t* slot = scratchpad + j * 16U;

    if constexpr (UseTTable)
        aes_single_round_ttable(slot, c, a, aes_tables);
    else
        aes_single_round(slot, c, a);
    reinterpret_cast<std::uint64_t*>(slot)[0] =
        reinterpret_cast<const std::uint64_t*>(c)[0] ^ reinterpret_cast<const std::uint64_t*>(b)[0];
    reinterpret_cast<std::uint64_t*>(slot)[1] =
        reinterpret_cast<const std::uint64_t*>(c)[1] ^ reinterpret_cast<const std::uint64_t*>(b)[1];
    variant1_mutate(slot);

    j = cn_index_masked(c, AddressMask);
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

    copy16(b, c);
}

template <std::uint8_t VariantIndex, int Unroll, bool UseTTable = false>
__device__ __forceinline__ void split_memory_loop_tuned(std::uint8_t* scratchpad,
                                                        const SplitContext& ctx,
                                                        const std::uint32_t* aes_tables = nullptr)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    static_assert(Unroll == 1 || Unroll == 2 || Unroll == 4, "unsupported CryptoNight loop unroll");
    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    static_assert((cfg.iterations % Unroll) == 0, "CryptoNight iteration count must divide by unroll");

    alignas(16) std::uint8_t a[16];
    alignas(16) std::uint8_t b[16];
    alignas(16) std::uint8_t c[16];
    alignas(16) std::uint8_t t[16];
    copy16(a, ctx.a);
    copy16(b, ctx.b);
    const std::uint64_t tweak = ctx.tweak;

    #pragma unroll 1
    for (std::uint32_t i = 0; i < cfg.iterations; i += Unroll) {
        #pragma unroll
        for (int u = 0; u < Unroll; ++u)
            split_memory_iteration<address_mask, UseTTable>(scratchpad, a, b, c, t, tweak, aes_tables);
    }
}

template <std::uint8_t VariantIndex, int Unroll, bool UseTTable = false>
__device__ __forceinline__ void split_memory_loop_dual_tuned(std::uint8_t* scratchpad0,
                                                             const SplitContext& ctx0,
                                                             std::uint8_t* scratchpad1,
                                                             const SplitContext& ctx1,
                                                             const std::uint32_t* aes_tables = nullptr)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    static_assert(Unroll == 1 || Unroll == 2 || Unroll == 4, "unsupported CryptoNight loop unroll");
    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    static_assert((cfg.iterations % Unroll) == 0, "CryptoNight iteration count must divide by unroll");

    alignas(16) std::uint8_t a0[16], b0[16], c0[16], t0[16];
    alignas(16) std::uint8_t a1[16], b1[16], c1[16], t1[16];
    copy16(a0, ctx0.a);
    copy16(b0, ctx0.b);
    copy16(a1, ctx1.a);
    copy16(b1, ctx1.b);
    const std::uint64_t tweak0 = ctx0.tweak;
    const std::uint64_t tweak1 = ctx1.tweak;

    #pragma unroll 1
    for (std::uint32_t i = 0; i < cfg.iterations; i += Unroll) {
        #pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            split_memory_iteration<address_mask, UseTTable>(scratchpad0, a0, b0, c0, t0, tweak0, aes_tables);
            split_memory_iteration<address_mask, UseTTable>(scratchpad1, a1, b1, c1, t1, tweak1, aes_tables);
        }
    }
}

template <std::uint8_t VariantIndex>
__device__ __forceinline__ void split_memory_loop(std::uint8_t* scratchpad,
                                                  const SplitContext& ctx)
{
    split_memory_loop_tuned<VariantIndex, 1, false>(scratchpad, ctx, nullptr);
}

template <std::uint8_t VariantIndex, bool UseTTable = false>
__device__ __forceinline__ void split_finalize(std::uint8_t* scratchpad,
                                               SplitContext& ctx,
                                               std::uint8_t out[32],
                                               const std::uint32_t* aes_tables = nullptr)
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
            if constexpr (UseTTable)
                aes_pseudo_round_ttable(x, expanded, aes_tables);
            else
                aes_pseudo_round(x, expanded);
        }
    }

    #pragma unroll
    for (int i = 0; i < 128; ++i) ctx.state[64 + i] = text[i];
    keccak_permute_state(ctx.state);
    dispatch_extra_hash(static_cast<std::uint8_t>(ctx.state[0] & 3U), ctx.state, out);
}

} // namespace yerbas::cuda::cryptonight
