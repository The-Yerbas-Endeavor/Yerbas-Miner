// Four-lane CN-Fast shuffle-first AES/T-table scheduling candidate.
//
// Keep the existing mode-444 kernel symbol so the hardened production selector
// can parity-test and benchmark this candidate against the proven mode-44
// T-table baseline without changing production routing or cache formats.
//
// This candidate deliberately keeps the same four-lane scratchpad layout,
// dependent-address path, serial 64x64 multiply/add path, and state ownership as
// the production 4-lane kernel. The only intended CN-Fast difference is AES
// scheduling: exchange the four source words first, then perform each lane's
// four shared-memory T-table lookups. Production currently does the opposite --
// four local table lookups first, then shuffles the computed contributions.
//
// This is a regression-recovery experiment: if the older shuffle-first ordering
// is materially faster, the previous local-read-first optimization was harmful
// on this architecture. Promotion still requires exact full-scratchpad parity
// and a measured win in the existing hardened selector.

namespace {

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_cg(
    std::size_t count,
    std::uint8_t* __restrict__ scratchpads,
    const cryptonight::SplitContext* __restrict__ contexts)
{
    __shared__ std::uint32_t tables[1024];
    cryptonight::init_shared_aes_ttables(tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    constexpr std::uint32_t byte_address_mask =
        static_cast<std::uint32_t>(address_mask << 4U);

    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~3U;
    const unsigned int group = global_thread >> 2U;
    if (group >= count) return;
    const unsigned int mask = 0xFU << base;

    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + 256;
    const std::uint32_t* t2 = tables + 512;
    const std::uint32_t* t3 = tables + 768;

    std::uint8_t* scratch =
        scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];
    std::uint32_t aw = cryptonight::load32le_aligned(ctx.a + lane * 4U);
    std::uint32_t bw = cryptonight::load32le_aligned(ctx.b + lane * 4U);

    std::uint32_t aw1_shadow = 0U;
    if (lane == 0U)
        aw1_shadow = cryptonight::load32le_aligned(ctx.a + 4U);

    const std::uint64_t tweak = ctx.tweak;
    const std::uint32_t tweak_lo = static_cast<std::uint32_t>(tweak);
    const std::uint32_t tweak_hi = static_cast<std::uint32_t>(tweak >> 32U);

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint32_t aw0 = __shfl_sync(mask, aw, base);
        const std::uint32_t aw1 = lane == 0U ? aw1_shadow : 0U;
        const std::uint32_t byte_offset = aw0 & byte_address_mask;
        std::uint8_t* slot = scratch + static_cast<std::size_t>(byte_offset);

        const std::uint32_t own =
            reinterpret_cast<const std::uint32_t*>(slot)[lane];

        // Shuffle-first AES schedule. For CN-Fast this is intentionally the
        // inverse ordering of production mode 44: exchange the source words
        // before issuing the four shared T-table reads.
        const std::uint32_t w1 =
            __shfl_sync(mask, own, base + ((lane + 1U) & 3U));
        const std::uint32_t w2 =
            __shfl_sync(mask, own, base + ((lane + 2U) & 3U));
        const std::uint32_t w3 =
            __shfl_sync(mask, own, base + ((lane + 3U) & 3U));
        const std::uint8_t x0 = static_cast<std::uint8_t>(own);
        const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
        const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
        const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);
        const std::uint32_t cw =
            t0[x0] ^ t1[x1] ^ t2[x2] ^ t3[x3] ^ aw;

        std::uint32_t first_store_word = cw ^ bw;
        if (lane == 2U) {
            const std::uint8_t tmp =
                static_cast<std::uint8_t>(first_store_word >> 24U);
            constexpr std::uint32_t table = 0x75310U;
            const std::uint8_t mutate_index = static_cast<std::uint8_t>(
                ((((tmp >> 3U) & 6U) | (tmp & 1U)) << 1U));
            first_store_word ^= ((table >> mutate_index) & 0x30U) << 24U;
        }
        reinterpret_cast<std::uint32_t*>(slot)[lane] = first_store_word;

        // Keep the production dependent-address/multiply path unchanged.
        const std::uint32_t c0 = __shfl_sync(mask, cw, base);
        const std::uint32_t byte_offset2 = c0 & byte_address_mask;
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(byte_offset2);

        const std::uint32_t t_word =
            reinterpret_cast<const std::uint32_t*>(slot2)[lane];
        const std::uint32_t t0_lo = lane == 0U ? t_word : 0U;
        const std::uint32_t c1 = __shfl_sync(mask, cw, base + 1U);
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t na0_lo = 0U;
        std::uint32_t na0_hi = 0U;
        std::uint32_t na1_lo = 0U;
        std::uint32_t na1_hi = 0U;
        if (lane == 0U) {
            const std::uint64_t c_low =
                static_cast<std::uint64_t>(c0) |
                (static_cast<std::uint64_t>(c1) << 32U);
            const std::uint64_t t0v =
                static_cast<std::uint64_t>(t0_lo) |
                (static_cast<std::uint64_t>(t0_hi) << 32U);
            const std::uint64_t lo = c_low * t0v;
            const std::uint64_t hi = __umul64hi(c_low, t0v);
            const std::uint64_t olda0 =
                static_cast<std::uint64_t>(aw0) |
                (static_cast<std::uint64_t>(aw1) << 32U);
            const std::uint64_t olda1 =
                static_cast<std::uint64_t>(old_a2) |
                (static_cast<std::uint64_t>(old_a3) << 32U);
            const std::uint64_t na0 = olda0 + hi;
            const std::uint64_t na1 = olda1 + lo;
            na0_lo = static_cast<std::uint32_t>(na0);
            na0_hi = static_cast<std::uint32_t>(na0 >> 32U);
            na1_lo = static_cast<std::uint32_t>(na1);
            na1_hi = static_cast<std::uint32_t>(na1 >> 32U);

            aw1_shadow = na0_hi ^ t0_hi;
        }

        na0_lo = __shfl_sync(mask, na0_lo, base);
        na0_hi = __shfl_sync(mask, na0_hi, base);
        na1_lo = __shfl_sync(mask, na1_lo, base);
        na1_hi = __shfl_sync(mask, na1_hi, base);

        const std::uint32_t store_word =
            lane == 0U ? na0_lo :
            lane == 1U ? na0_hi :
            lane == 2U ? (na1_lo ^ tweak_lo) : (na1_hi ^ tweak_hi);
        reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

        const std::uint32_t na_word =
            lane == 0U ? na0_lo :
            lane == 1U ? na0_hi :
            lane == 2U ? na1_lo : na1_hi;
        aw = na_word ^ t_word;
        bw = cw;
    }
}

} // anonymous namespace
