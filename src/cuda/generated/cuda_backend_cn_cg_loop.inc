// Four-lane split-multiply CryptoNight phase-2 candidate.
//
// Keep the existing mode-444 kernel symbol so the hardened production selector
// can parity-test and benchmark this candidate against the proven mode-44
// T-table baseline without changing production routing or cache formats.
//
// The AES/T-table and direct scratchpad path remains four lanes per hash. Only
// the dependency-heavy 64x64 multiply is changed: lane 0 computes the high half
// while lane 1 computes the low half concurrently. The resulting A words stay
// owned by those lanes and are shuffled only to the lanes that consume them.
//
// Promotion still requires exact full-scratchpad parity and a measured >=5%
// real-production-batch win in the existing selector.

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

    // The next dependent address uses the low A word. Keep lane 0's copy of
    // A[1] locally so fetching it never extends that address dependency chain.
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
        std::uint32_t cw = 0U;

        if constexpr (VariantIndex == 2) {
            // Keep the proven CN-Fast table schedule: issue each lane's four
            // independent T-table reads before exchanging ShiftRows terms.
            const std::uint32_t q0 = t0[static_cast<std::uint8_t>(own)];
            const std::uint32_t q1_local =
                t1[static_cast<std::uint8_t>(own >> 8U)];
            const std::uint32_t q2_local =
                t2[static_cast<std::uint8_t>(own >> 16U)];
            const std::uint32_t q3_local =
                t3[static_cast<std::uint8_t>(own >> 24U)];
            const std::uint32_t q1 =
                __shfl_sync(mask, q1_local, base + ((lane + 1U) & 3U));
            const std::uint32_t q2 =
                __shfl_sync(mask, q2_local, base + ((lane + 2U) & 3U));
            const std::uint32_t q3 =
                __shfl_sync(mask, q3_local, base + ((lane + 3U) & 3U));
            cw = q0 ^ q1 ^ q2 ^ q3 ^ aw;
        } else {
            const std::uint32_t w1 =
                __shfl_sync(mask, own, base + ((lane + 1U) & 3U));
            const std::uint32_t w2 =
                __shfl_sync(mask, own, base + ((lane + 2U) & 3U));
            const std::uint32_t w3 =
                __shfl_sync(mask, own, base + ((lane + 3U) & 3U));
            cw = t0[static_cast<std::uint8_t>(own)] ^
                 t1[static_cast<std::uint8_t>(w1 >> 8U)] ^
                 t2[static_cast<std::uint8_t>(w2 >> 16U)] ^
                 t3[static_cast<std::uint8_t>(w3 >> 24U)] ^ aw;
        }

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

        const std::uint32_t c0 = __shfl_sync(mask, cw, base);
        const std::uint32_t byte_offset2 = c0 & byte_address_mask;
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(byte_offset2);
        const std::uint32_t t_word =
            reinterpret_cast<const std::uint32_t*>(slot2)[lane];

        // Both multiply lanes need the same 64-bit operands. This adds one
        // t0-low shuffle relative to the baseline, but removes one serialized
        // 64-bit multiply from lane 0's critical dependency chain.
        const std::uint32_t c1 = __shfl_sync(mask, cw, base + 1U);
        const std::uint32_t t0_lo = __shfl_sync(mask, t_word, base);
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t na0_lo_local = 0U;
        std::uint32_t na0_hi_local = 0U;
        std::uint32_t na1_lo_local = 0U;
        std::uint32_t na1_hi_local = 0U;

        if (lane == 0U || lane == 1U) {
            const std::uint64_t c_low =
                static_cast<std::uint64_t>(c0) |
                (static_cast<std::uint64_t>(c1) << 32U);
            const std::uint64_t t0v =
                static_cast<std::uint64_t>(t0_lo) |
                (static_cast<std::uint64_t>(t0_hi) << 32U);

            if (lane == 0U) {
                const std::uint64_t hi = __umul64hi(c_low, t0v);
                const std::uint64_t olda0 =
                    static_cast<std::uint64_t>(aw0) |
                    (static_cast<std::uint64_t>(aw1) << 32U);
                const std::uint64_t na0 = olda0 + hi;
                na0_lo_local = static_cast<std::uint32_t>(na0);
                na0_hi_local = static_cast<std::uint32_t>(na0 >> 32U);
                aw1_shadow = na0_hi_local ^ t0_hi;
            } else {
                const std::uint64_t lo = c_low * t0v;
                const std::uint64_t olda1 =
                    static_cast<std::uint64_t>(old_a2) |
                    (static_cast<std::uint64_t>(old_a3) << 32U);
                const std::uint64_t na1 = olda1 + lo;
                na1_lo_local = static_cast<std::uint32_t>(na1);
                na1_hi_local = static_cast<std::uint32_t>(na1 >> 32U);
            }
        }

        // Keep each result close to its producer. Lane 0 consumes na0_lo
        // locally; lane 1 consumes na0_hi from lane 0; lanes 2/3 consume the
        // low/high halves produced by lane 1.
        const std::uint32_t na0_hi =
            __shfl_sync(mask, na0_hi_local, base);
        const std::uint32_t na1_lo =
            __shfl_sync(mask, na1_lo_local, base + 1U);
        const std::uint32_t na1_hi =
            __shfl_sync(mask, na1_hi_local, base + 1U);

        const std::uint32_t na_word =
            lane == 0U ? na0_lo_local :
            lane == 1U ? na0_hi :
            lane == 2U ? na1_lo : na1_hi;

        const std::uint32_t store_word =
            lane == 2U ? (na_word ^ tweak_lo) :
            lane == 3U ? (na_word ^ tweak_hi) : na_word;
        reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

        aw = na_word ^ t_word;
        bw = cw;
    }
}

} // anonymous namespace
