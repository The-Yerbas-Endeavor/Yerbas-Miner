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

// CryptoNight's pseudo AES pass consumes only the first ten 16-byte round
// keys.  The generic AES-256 expansion builds 240 bytes; producing only the
// 160 bytes actually consumed by setup/collapse cuts key-schedule work and
// reduces per-thread local state in these two kernels.
__device__ __forceinline__ void cn_expand_key_10rounds(const std::uint8_t key[32],
                                                       std::uint8_t expanded[160])
{
    #pragma unroll
    for (int i = 0; i < 32; ++i) expanded[i] = key[i];

    std::uint8_t rcon = 1;
    int generated = 32;
    while (generated < 160) {
        std::uint8_t t[4] = {
            expanded[generated - 4], expanded[generated - 3],
            expanded[generated - 2], expanded[generated - 1]
        };
        if ((generated & 31) == 0) {
            const std::uint8_t x = t[0];
            t[0] = static_cast<std::uint8_t>(aes_sbox(t[1]) ^ rcon);
            t[1] = aes_sbox(t[2]);
            t[2] = aes_sbox(t[3]);
            t[3] = aes_sbox(x);
            rcon = xtime(rcon);
        } else if ((generated & 31) == 16) {
            t[0] = aes_sbox(t[0]);
            t[1] = aes_sbox(t[1]);
            t[2] = aes_sbox(t[2]);
            t[3] = aes_sbox(t[3]);
        }
        #pragma unroll
        for (int i = 0; i < 4 && generated < 160; ++i) {
            expanded[generated] = static_cast<std::uint8_t>(expanded[generated - 32] ^ t[i]);
            ++generated;
        }
    }
}

template <std::uint8_t VariantIndex>
__device__ __forceinline__ bool split_setup(const std::uint8_t* input,
                                            std::size_t length,
                                            std::uint8_t* __restrict__ scratchpad,
                                            SplitContext& ctx)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    if (scratchpad == nullptr || input == nullptr || length < 43) return false;

    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;

    keccak1600(input, length, ctx.state);

    alignas(16) std::uint8_t text[128];
    #pragma unroll
    for (int block = 0; block < 8; ++block)
        copy16(text + block * 16, ctx.state + 64 + block * 16);

    alignas(16) std::uint8_t expanded[160];
    cn_expand_key_10rounds(ctx.state, expanded);

    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block)
            aes_pseudo_round(text + block * 16, expanded);

        // Both buffers are 16-byte aligned.  Store the 128-byte scratchpad row
        // as sixteen native 64-bit words instead of 128 byte stores.
        #pragma unroll
        for (int block = 0; block < 8; ++block)
            copy16(scratchpad + i * 128U + static_cast<std::size_t>(block * 16),
                   text + block * 16);
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

// The memory-hard loop dominates CryptoNight runtime.  The byte-array version
// kept a/b/c/t as indexable local arrays; NVCC can spill those to per-thread
// local memory, adding extra global traffic around every dependent scratchpad
// access.  Keep the complete hot state in six 64-bit registers instead.
__device__ __forceinline__ std::uint8_t cn_byte64(std::uint64_t v, unsigned int byte)
{
    return static_cast<std::uint8_t>(v >> (byte * 8U));
}

__device__ __forceinline__ std::uint32_t cn_aes_mix_column(std::uint8_t x0,
                                                           std::uint8_t x1,
                                                           std::uint8_t x2,
                                                           std::uint8_t x3,
                                                           std::uint32_t key)
{
    const std::uint8_t t = static_cast<std::uint8_t>(x0 ^ x1 ^ x2 ^ x3);
    const std::uint8_t o0 = static_cast<std::uint8_t>(x0 ^ t ^ xtime(static_cast<std::uint8_t>(x0 ^ x1)));
    const std::uint8_t o1 = static_cast<std::uint8_t>(x1 ^ t ^ xtime(static_cast<std::uint8_t>(x1 ^ x2)));
    const std::uint8_t o2 = static_cast<std::uint8_t>(x2 ^ t ^ xtime(static_cast<std::uint8_t>(x2 ^ x3)));
    const std::uint8_t o3 = static_cast<std::uint8_t>(x3 ^ t ^ xtime(static_cast<std::uint8_t>(x3 ^ x0)));
    const std::uint32_t mixed = static_cast<std::uint32_t>(o0)
        | (static_cast<std::uint32_t>(o1) << 8)
        | (static_cast<std::uint32_t>(o2) << 16)
        | (static_cast<std::uint32_t>(o3) << 24);
    return mixed ^ key;
}

__device__ __forceinline__ void cn_aes_single_round_u64(std::uint64_t in0,
                                                        std::uint64_t in1,
                                                        std::uint64_t key0,
                                                        std::uint64_t key1,
                                                        std::uint64_t& out0,
                                                        std::uint64_t& out1)
{
    // SubBytes + ShiftRows, grouped directly into the four output columns.
    const std::uint32_t c0 = cn_aes_mix_column(
        aes_sbox(cn_byte64(in0, 0)), aes_sbox(cn_byte64(in0, 5)),
        aes_sbox(cn_byte64(in1, 2)), aes_sbox(cn_byte64(in1, 7)),
        static_cast<std::uint32_t>(key0));
    const std::uint32_t c1 = cn_aes_mix_column(
        aes_sbox(cn_byte64(in0, 4)), aes_sbox(cn_byte64(in1, 1)),
        aes_sbox(cn_byte64(in1, 6)), aes_sbox(cn_byte64(in0, 3)),
        static_cast<std::uint32_t>(key0 >> 32));
    const std::uint32_t c2 = cn_aes_mix_column(
        aes_sbox(cn_byte64(in1, 0)), aes_sbox(cn_byte64(in1, 5)),
        aes_sbox(cn_byte64(in0, 2)), aes_sbox(cn_byte64(in0, 7)),
        static_cast<std::uint32_t>(key1));
    const std::uint32_t c3 = cn_aes_mix_column(
        aes_sbox(cn_byte64(in1, 4)), aes_sbox(cn_byte64(in0, 1)),
        aes_sbox(cn_byte64(in0, 6)), aes_sbox(cn_byte64(in1, 3)),
        static_cast<std::uint32_t>(key1 >> 32));

    out0 = static_cast<std::uint64_t>(c0) | (static_cast<std::uint64_t>(c1) << 32);
    out1 = static_cast<std::uint64_t>(c2) | (static_cast<std::uint64_t>(c3) << 32);
}

__device__ __forceinline__ std::uint64_t cn_variant1_mutate_hi(std::uint64_t word)
{
    // block[11] is byte 3 of the upper 64-bit word.
    const std::uint8_t tmp = static_cast<std::uint8_t>(word >> 24);
    constexpr std::uint32_t table = 0x75310U;
    const std::uint8_t index = static_cast<std::uint8_t>((((tmp >> 3) & 6U) | (tmp & 1U)) << 1);
    const std::uint8_t mutated = static_cast<std::uint8_t>(tmp ^ ((table >> index) & 0x30U));
    constexpr std::uint64_t mask = ~(std::uint64_t{0xff} << 24);
    return (word & mask) | (static_cast<std::uint64_t>(mutated) << 24);
}

template <std::uint8_t VariantIndex>
__device__ __forceinline__ void split_memory_loop(std::uint8_t* __restrict__ scratchpad,
                                                  const SplitContext& ctx)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;

    std::uint64_t a0 = cn_load64_aligned(ctx.a);
    std::uint64_t a1 = cn_load64_aligned(ctx.a + 8);
    std::uint64_t b0 = cn_load64_aligned(ctx.b);
    std::uint64_t b1 = cn_load64_aligned(ctx.b + 8);
    const std::uint64_t tweak = ctx.tweak;

    #pragma unroll 1
    for (std::uint32_t i = 0; i < cfg.iterations; ++i) {
        std::size_t j = static_cast<std::size_t>((a0 >> 4) & address_mask);
        std::uint64_t* slot = reinterpret_cast<std::uint64_t*>(scratchpad + j * 16U);

        const std::uint64_t in0 = slot[0];
        const std::uint64_t in1 = slot[1];
        std::uint64_t c0;
        std::uint64_t c1;
        cn_aes_single_round_u64(in0, in1, a0, a1, c0, c1);

        slot[0] = c0 ^ b0;
        slot[1] = cn_variant1_mutate_hi(c1 ^ b1);

        j = static_cast<std::size_t>((c0 >> 4) & address_mask);
        slot = reinterpret_cast<std::uint64_t*>(scratchpad + j * 16U);
        const std::uint64_t t0 = slot[0];
        const std::uint64_t t1 = slot[1];

        const std::uint64_t lo = c0 * t0;
        const std::uint64_t hi = __umul64hi(c0, t0);

        a0 += hi;
        a1 += lo;
        slot[0] = a0;
        slot[1] = a1 ^ tweak;

        a0 ^= t0;
        a1 ^= t1;
        b0 = c0;
        b1 = c1;
    }
}

template <std::uint8_t VariantIndex>
__device__ __forceinline__ void split_finalize(std::uint8_t* __restrict__ scratchpad,
                                               SplitContext& ctx,
                                               std::uint8_t out[32])
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    constexpr VariantConfig cfg = config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;

    alignas(16) std::uint8_t text[128];
    #pragma unroll
    for (int block = 0; block < 8; ++block)
        copy16(text + block * 16, ctx.state + 64 + block * 16);

    alignas(16) std::uint8_t expanded[160];
    cn_expand_key_10rounds(ctx.state + 32, expanded);
    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block) {
            std::uint8_t* x = text + block * 16;
            xor16(x, scratchpad + i * 128U + static_cast<std::size_t>(block * 16));
            aes_pseudo_round(x, expanded);
        }
    }

    #pragma unroll
    for (int block = 0; block < 8; ++block)
        copy16(ctx.state + 64 + block * 16, text + block * 16);

    keccak_permute_state(ctx.state);
    dispatch_extra_hash(static_cast<std::uint8_t>(ctx.state[0] & 3U), ctx.state, out);
}

} // namespace yerbas::cuda::cryptonight
