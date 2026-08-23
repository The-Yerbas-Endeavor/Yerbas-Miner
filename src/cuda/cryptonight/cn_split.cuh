#pragma once

#include "cuda/cryptonight/cn_slow_hash.cuh"

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::cryptonight {

struct alignas(16) CnState128 {
    std::uint64_t lo;
    std::uint64_t hi;
};

__device__ __forceinline__ std::uint8_t* cn_state_bytes(CnState128& v)
{
    return reinterpret_cast<std::uint8_t*>(&v);
}

__device__ __forceinline__ const std::uint8_t* cn_state_bytes(const CnState128& v)
{
    return reinterpret_cast<const std::uint8_t*>(&v);
}

struct alignas(16) SplitContext {
    std::uint8_t state[200];
    CnState128 a;
    CnState128 b;
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

    // Keep the hot-loop state in native 64-bit words. All source offsets are
    // naturally aligned inside the 200-byte Keccak state.
    ctx.a.lo = cn_load64_aligned(ctx.state) ^ cn_load64_aligned(ctx.state + 32);
    ctx.a.hi = cn_load64_aligned(ctx.state + 8) ^ cn_load64_aligned(ctx.state + 40);
    ctx.b.lo = cn_load64_aligned(ctx.state + 16) ^ cn_load64_aligned(ctx.state + 48);
    ctx.b.hi = cn_load64_aligned(ctx.state + 24) ^ cn_load64_aligned(ctx.state + 56);

    ctx.tweak = cn_load64(input + 35) ^ cn_load64_aligned(ctx.state + 192);
    return true;
}

template <std::size_t AddressMask, bool UseTTable = false>
__device__ __forceinline__ void split_memory_iteration_native(std::uint8_t* scratchpad,
                                                               CnState128& a,
                                                               CnState128& b,
                                                               CnState128& c,
                                                               std::uint64_t tweak,
                                                               const std::uint32_t* aes_tables = nullptr)
{
    std::size_t j = static_cast<std::size_t>((a.lo >> 4) & AddressMask);
    std::uint8_t* slot = scratchpad + j * 16U;

    if constexpr (UseTTable)
        aes_single_round_ttable(slot, cn_state_bytes(c), cn_state_bytes(a), aes_tables);
    else
        aes_single_round(slot, cn_state_bytes(c), cn_state_bytes(a));

    auto* slot64 = reinterpret_cast<std::uint64_t*>(slot);
    slot64[0] = c.lo ^ b.lo;
    slot64[1] = c.hi ^ b.hi;
    variant1_mutate(slot);

    j = static_cast<std::size_t>((c.lo >> 4) & AddressMask);
    slot = scratchpad + j * 16U;
    slot64 = reinterpret_cast<std::uint64_t*>(slot);

    const std::uint64_t t0 = slot64[0];
    const std::uint64_t t1 = slot64[1];
    const std::uint64_t lo = c.lo * t0;
    const std::uint64_t hi = __umul64hi(c.lo, t0);

    std::uint64_t a0 = a.lo + hi;
    std::uint64_t a1 = a.hi + lo;
    slot64[0] = a0;
    slot64[1] = a1 ^ tweak;

    a.lo = a0 ^ t0;
    a.hi = a1 ^ t1;
    b = c;
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

    CnState128 a = ctx.a;
    CnState128 b = ctx.b;
    CnState128 c{};
    const std::uint64_t tweak = ctx.tweak;

    #pragma unroll 1
    for (std::uint32_t i = 0; i < cfg.iterations; i += Unroll) {
        #pragma unroll
        for (int u = 0; u < Unroll; ++u)
            split_memory_iteration_native<address_mask, UseTTable>(scratchpad, a, b, c, tweak, aes_tables);
    }
}

// Interleave two independent CryptoNight states in one CUDA thread. Each hash
// preserves the exact canonical iteration order; the only change is exposing
// independent work from the second hash while the first hash follows its
// data-dependent scratchpad chain. Production selects this path only after
// parity validation and an empirical per-device/per-variant benchmark.
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

    CnState128 a0 = ctx0.a;
    CnState128 b0 = ctx0.b;
    CnState128 c0{};
    CnState128 a1 = ctx1.a;
    CnState128 b1 = ctx1.b;
    CnState128 c1{};
    const std::uint64_t tweak0 = ctx0.tweak;
    const std::uint64_t tweak1 = ctx1.tweak;

    #pragma unroll 1
    for (std::uint32_t i = 0; i < cfg.iterations; i += Unroll) {
        #pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            split_memory_iteration_native<address_mask, UseTTable>(scratchpad0, a0, b0, c0, tweak0, aes_tables);
            split_memory_iteration_native<address_mask, UseTTable>(scratchpad1, a1, b1, c1, tweak1, aes_tables);
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
