// Generic cooperative CryptoNight v4 candidate.
//
// v4 keeps the proven v3 register-distributed state/indexing semantics but
// changes the first scratchpad fetch. Each lane performs one naturally aligned
// 32-bit load from the 16-byte slot, then sub-warp shuffles assemble the four
// AES column bytes. This replaces four scattered byte loads per lane with one
// coalesced word load per lane. No GPU model assumptions are used.

namespace {

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_coop4_v4(std::size_t count,
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

        // Four participating lanes fetch the complete 16-byte slot as four
        // aligned words. AES columns are then formed from rotated source lanes.
        const std::uint32_t packed = reinterpret_cast<const std::uint32_t*>(slot)[lane];
        const unsigned int s0 = base + lane;
        const unsigned int s1 = base + ((lane + 1U) & 3U);
        const unsigned int s2 = base + ((lane + 2U) & 3U);
        const unsigned int s3 = base + ((lane + 3U) & 3U);
        const std::uint32_t w0 = __shfl_sync(mask, packed, s0);
        const std::uint32_t w1 = __shfl_sync(mask, packed, s1);
        const std::uint32_t w2 = __shfl_sync(mask, packed, s2);
        const std::uint32_t w3 = __shfl_sync(mask, packed, s3);
        const std::uint8_t x0 = static_cast<std::uint8_t>(w0);
        const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
        const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
        const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);

        const std::uint32_t cw = coop_mix_column(
            cryptonight::aes_sbox(x0), cryptonight::aes_sbox(x1),
            cryptonight::aes_sbox(x2), cryptonight::aes_sbox(x3)) ^ aw;

        reinterpret_cast<std::uint32_t*>(slot)[lane] = cw ^ bw;
        __syncwarp(mask);
        if (lane == 0U) cryptonight::variant1_mutate(slot);

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
        const std::uint32_t out_a1 = __shfl_sync(mask, next_a1, base + 1U);
        const std::uint32_t out_a2 = __shfl_sync(mask, next_a2, base + 2U);
        const std::uint32_t out_a3 = __shfl_sync(mask, next_a3, base + 3U);
        aw = lane == 0U ? out_a0 : lane == 1U ? out_a1 : lane == 2U ? out_a2 : out_a3;
        bw = cw;
        __syncwarp(mask);
    }
}

} // anonymous namespace
