// Generic CN-Fast cooperative v6 candidate.
//
// v6 targets the write+variant-mutation section measured at ~31% of the inner
// loop. The Variant-1 mutation changes only byte 11 of the just-produced
// 16-byte scratchpad value. In coop4 that byte belongs to lane 2 (word bytes
// 8..11), so mutate the outgoing lane-2 word in a register before the store.
// This removes the later lane-0 byte read/modify/write while preserving the
// mandatory sub-warp synchronization before the dependent second access.
// No GPU model assumptions are used.

namespace {

__device__ __forceinline__ std::uint32_t cn_variant1_mutate_word2(std::uint32_t word)
{
    const std::uint8_t tmp = static_cast<std::uint8_t>(word >> 24U); // byte 11
    constexpr std::uint32_t table = 0x75310U;
    const std::uint8_t index = static_cast<std::uint8_t>((((tmp >> 3U) & 6U) | (tmp & 1U)) << 1U);
    const std::uint8_t mutated = static_cast<std::uint8_t>(tmp ^ ((table >> index) & 0x30U));
    return (word & 0x00ffffffU) | (static_cast<std::uint32_t>(mutated) << 24U);
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_coop4_v6(std::size_t count,
                                                std::uint8_t* scratchpads,
                                                const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    const unsigned int gt = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int wl = threadIdx.x & 31U;
    const unsigned int base = wl & ~3U;
    const unsigned int group = gt >> 2;
    if (group >= count) return;
    const unsigned int mask = 0xFU << base;

    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];
    std::uint32_t aw = cryptonight::load32le_aligned(ctx.a + lane * 4U);
    std::uint32_t bw = cryptonight::load32le_aligned(ctx.b + lane * 4U);
    const std::uint64_t tweak = ctx.tweak;

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint32_t aw0 = __shfl_sync(mask, aw, base);
        const std::uint32_t aw1 = __shfl_sync(mask, aw, base + 1U);
        const std::uint64_t a_low = static_cast<std::uint64_t>(aw0) |
                                    (static_cast<std::uint64_t>(aw1) << 32U);
        const std::uint32_t index = static_cast<std::uint32_t>((a_low >> 4U) & address_mask);
        std::uint8_t* slot = scratch + static_cast<std::size_t>(index) * 16U;

        std::uint8_t x0, x1, x2, x3;
        if (lane == 0U) { x0=slot[0];  x1=slot[5];  x2=slot[10]; x3=slot[15]; }
        else if (lane == 1U) { x0=slot[4];  x1=slot[9];  x2=slot[14]; x3=slot[3]; }
        else if (lane == 2U) { x0=slot[8];  x1=slot[13]; x2=slot[2];  x3=slot[7]; }
        else { x0=slot[12]; x1=slot[1];  x2=slot[6];  x3=slot[11]; }

        const std::uint32_t cw = coop_mix_column(
            cryptonight::aes_sbox(x0), cryptonight::aes_sbox(x1),
            cryptonight::aes_sbox(x2), cryptonight::aes_sbox(x3)) ^ aw;

        std::uint32_t store_word = cw ^ bw;
        if (lane == 2U) store_word = cn_variant1_mutate_word2(store_word);
        reinterpret_cast<std::uint32_t*>(slot)[lane] = store_word;
        __syncwarp(mask);

        const std::uint32_t c0 = __shfl_sync(mask, cw, base);
        const std::uint32_t c1 = __shfl_sync(mask, cw, base + 1U);
        const std::uint32_t old_a0 = aw0;
        const std::uint32_t old_a1 = aw1;
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t next_a0 = 0U, next_a1 = 0U, next_a2 = 0U, next_a3 = 0U;
        if (lane == 0U) {
            const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                        (static_cast<std::uint64_t>(c1) << 32U);
            const std::uint32_t j = static_cast<std::uint32_t>((c_low >> 4U) & address_mask);
            std::uint8_t* slot2 = scratch + static_cast<std::size_t>(j) * 16U;
            const std::uint64_t t0 = cryptonight::cn_load64_aligned(slot2);
            const std::uint64_t t1 = cryptonight::cn_load64_aligned(slot2 + 8);
            const std::uint64_t lo = c_low * t0;
            const std::uint64_t hi = __umul64hi(c_low, t0);
            const std::uint64_t olda0 = static_cast<std::uint64_t>(old_a0) |
                                        (static_cast<std::uint64_t>(old_a1) << 32U);
            const std::uint64_t olda1 = static_cast<std::uint64_t>(old_a2) |
                                        (static_cast<std::uint64_t>(old_a3) << 32U);
            const std::uint64_t na0 = olda0 + hi;
            const std::uint64_t na1 = olda1 + lo;
            cryptonight::cn_store64_aligned(slot2, na0);
            cryptonight::cn_store64_aligned(slot2 + 8, na1 ^ tweak);
            const std::uint64_t xa0 = na0 ^ t0;
            const std::uint64_t xa1 = na1 ^ t1;
            next_a0 = static_cast<std::uint32_t>(xa0);
            next_a1 = static_cast<std::uint32_t>(xa0 >> 32U);
            next_a2 = static_cast<std::uint32_t>(xa1);
            next_a3 = static_cast<std::uint32_t>(xa1 >> 32U);
        }

        const std::uint32_t out_a0 = __shfl_sync(mask, next_a0, base);
        const std::uint32_t out_a1 = __shfl_sync(mask, next_a1, base);
        const std::uint32_t out_a2 = __shfl_sync(mask, next_a2, base);
        const std::uint32_t out_a3 = __shfl_sync(mask, next_a3, base);
        aw = lane == 0U ? out_a0 : lane == 1U ? out_a1 : lane == 2U ? out_a2 : out_a3;
        bw = cw;
        __syncwarp(mask);
    }
}

} // anonymous namespace
