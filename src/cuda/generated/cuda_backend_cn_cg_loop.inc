// Cooperative-multiply CryptoNight phase-2 candidate.
//
// Keep the historical mode-444 kernel symbol so the existing hardened selector
// can parity-test and benchmark this candidate without changing production
// routing or cache formats.  The old .cg scratchpad experiment is replaced here
// with ordinary coalesced global accesses plus a four-lane 32x32 partial-product
// implementation of the dependency-heavy 64x64->128 multiply.
//
// Each lane computes one 32x32->64 partial product concurrently:
//   lane 0: c.lo * t.lo
//   lane 1: c.lo * t.hi
//   lane 2: c.hi * t.lo
//   lane 3: c.hi * t.hi
// Lane 0 reconstructs the exact 128-bit product from shuffled 32-bit limbs.
// The existing selector requires exact full-scratchpad parity and a measured
// production-batch win before this kernel can be promoted.

namespace {

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_cg(std::size_t count,
                                                  std::uint8_t* scratchpads,
                                                  const cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t tables[1024];
    cryptonight::init_shared_aes_ttables(tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~3U;
    const unsigned int group = global_thread >> 2U;
    if (group >= count) return;
    const unsigned int mask = 0xFU << base;

    const std::uint32_t* table0 = tables;
    const std::uint32_t* table1 = tables + 256;
    const std::uint32_t* table2 = tables + 512;
    const std::uint32_t* table3 = tables + 768;

    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];
    std::uint32_t aw = cryptonight::load32le_aligned(ctx.a + lane * 4U);
    std::uint32_t bw = cryptonight::load32le_aligned(ctx.b + lane * 4U);
    const std::uint64_t tweak = ctx.tweak;
    const std::uint32_t tweak_lo = static_cast<std::uint32_t>(tweak);
    const std::uint32_t tweak_hi = static_cast<std::uint32_t>(tweak >> 32U);

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint32_t aw0 = __shfl_sync(mask, aw, base);
        const std::uint32_t aw1 = __shfl_sync(mask, aw, base + 1U);
        const std::uint64_t a_low = static_cast<std::uint64_t>(aw0) |
                                    (static_cast<std::uint64_t>(aw1) << 32U);
        const std::uint32_t index = static_cast<std::uint32_t>((a_low >> 4U) & address_mask);
        std::uint8_t* slot = scratch + static_cast<std::size_t>(index) * 16U;
        auto* slot_words = reinterpret_cast<std::uint32_t*>(slot);

        const std::uint32_t own = slot_words[lane];
        const std::uint32_t w1 = __shfl_sync(mask, own, base + ((lane + 1U) & 3U));
        const std::uint32_t w2 = __shfl_sync(mask, own, base + ((lane + 2U) & 3U));
        const std::uint32_t w3 = __shfl_sync(mask, own, base + ((lane + 3U) & 3U));
        const std::uint8_t x0 = static_cast<std::uint8_t>(own);
        const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
        const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
        const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);
        const std::uint32_t cw = table0[x0] ^ table1[x1] ^ table2[x2] ^ table3[x3] ^ aw;

        slot_words[lane] = cw ^ bw;
        __syncwarp(mask);
        if (lane == 0U) cryptonight::variant1_mutate(slot);
        __syncwarp(mask);

        const std::uint32_t c0 = __shfl_sync(mask, cw, base);
        const std::uint32_t c1 = __shfl_sync(mask, cw, base + 1U);
        const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                    (static_cast<std::uint64_t>(c1) << 32U);
        const std::uint32_t j = static_cast<std::uint32_t>((c_low >> 4U) & address_mask);
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(j) * 16U;
        auto* slot2_words = reinterpret_cast<std::uint32_t*>(slot2);

        const std::uint32_t t_word = slot2_words[lane];
        const std::uint32_t t0_lo = __shfl_sync(mask, t_word, base);
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        // Four 32x32 products replace the serialized lane-0 64x64 multiply.
        const std::uint32_t mul_a = lane < 2U ? c0 : c1;
        const std::uint32_t mul_b = (lane & 1U) == 0U ? t0_lo : t0_hi;
        const std::uint64_t partial = static_cast<std::uint64_t>(mul_a) *
                                      static_cast<std::uint64_t>(mul_b);
        const std::uint32_t partial_lo = static_cast<std::uint32_t>(partial);
        const std::uint32_t partial_hi = static_cast<std::uint32_t>(partial >> 32U);

        const std::uint32_t p00_lo = __shfl_sync(mask, partial_lo, base);
        const std::uint32_t p00_hi = __shfl_sync(mask, partial_hi, base);
        const std::uint32_t p01_lo = __shfl_sync(mask, partial_lo, base + 1U);
        const std::uint32_t p01_hi = __shfl_sync(mask, partial_hi, base + 1U);
        const std::uint32_t p10_lo = __shfl_sync(mask, partial_lo, base + 2U);
        const std::uint32_t p10_hi = __shfl_sync(mask, partial_hi, base + 2U);
        const std::uint32_t p11_lo = __shfl_sync(mask, partial_lo, base + 3U);
        const std::uint32_t p11_hi = __shfl_sync(mask, partial_hi, base + 3U);

        std::uint32_t na0_lo = 0U, na0_hi = 0U, na1_lo = 0U, na1_hi = 0U;
        if (lane == 0U) {
            // Exact unsigned 64x64->128 reconstruction from four 32-bit limbs.
            const std::uint64_t middle = static_cast<std::uint64_t>(p00_hi) +
                                         static_cast<std::uint64_t>(p01_lo) +
                                         static_cast<std::uint64_t>(p10_lo);
            const std::uint64_t lo = static_cast<std::uint64_t>(p00_lo) |
                                     (static_cast<std::uint64_t>(static_cast<std::uint32_t>(middle)) << 32U);
            const std::uint64_t p11 = static_cast<std::uint64_t>(p11_lo) |
                                      (static_cast<std::uint64_t>(p11_hi) << 32U);
            const std::uint64_t hi = p11 +
                                     static_cast<std::uint64_t>(p01_hi) +
                                     static_cast<std::uint64_t>(p10_hi) +
                                     (middle >> 32U);

            const std::uint64_t olda0 = static_cast<std::uint64_t>(aw0) |
                                        (static_cast<std::uint64_t>(aw1) << 32U);
            const std::uint64_t olda1 = static_cast<std::uint64_t>(old_a2) |
                                        (static_cast<std::uint64_t>(old_a3) << 32U);
            const std::uint64_t na0 = olda0 + hi;
            const std::uint64_t na1 = olda1 + lo;
            na0_lo = static_cast<std::uint32_t>(na0);
            na0_hi = static_cast<std::uint32_t>(na0 >> 32U);
            na1_lo = static_cast<std::uint32_t>(na1);
            na1_hi = static_cast<std::uint32_t>(na1 >> 32U);
        }
        na0_lo = __shfl_sync(mask, na0_lo, base);
        na0_hi = __shfl_sync(mask, na0_hi, base);
        na1_lo = __shfl_sync(mask, na1_lo, base);
        na1_hi = __shfl_sync(mask, na1_hi, base);

        const std::uint32_t store_word = lane == 0U ? na0_lo :
                                         lane == 1U ? na0_hi :
                                         lane == 2U ? (na1_lo ^ tweak_lo) : (na1_hi ^ tweak_hi);
        slot2_words[lane] = store_word;

        const std::uint32_t na_word = lane == 0U ? na0_lo :
                                      lane == 1U ? na0_hi :
                                      lane == 2U ? na1_lo : na1_hi;
        aw = na_word ^ t_word;
        bw = cw;
        __syncwarp(mask);
    }
}

} // anonymous namespace
