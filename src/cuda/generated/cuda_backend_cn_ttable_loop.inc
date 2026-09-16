// Mature-style CryptoNight phase-2 candidates.
//
// The established cooperative loop performs byte S-box lookups plus MixColumns
// arithmetic in the dependency-heavy inner loop. Mature CUDA CryptoNight miners
// instead keep the four AES T-tables in shared memory. These candidates adopt
// that proven execution model while preserving Yerbas' exact CN state machine.
//
//  - 2-lane: one contiguous 64-bit scratchpad load per lane.
//  - 4-lane: one contiguous 32-bit load per lane; ShiftRows bytes are exchanged
//    through subgroup shuffles instead of four scattered byte loads per lane.
//
// Experimental branch note: the 4-lane kernel below carries two independent
// hash states per four-lane subgroup. It issues the dependent scratchpad loads
// for hash A and hash B back-to-back before consuming either result. The goal is
// to expose instruction-level memory/multiply latency hiding without changing
// the CryptoNight state machine. The ordinary production branch is untouched.
//
// Neither path is trusted by construction: the production selector validates
// finalized output against cryptonight_loop_stage_cooperative before promotion.

namespace {

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable2(std::size_t count,
                                               std::uint8_t* __restrict__ scratchpads,
                                               const cryptonight::SplitContext* __restrict__ contexts)
{
    __shared__ std::uint32_t tables[1024];
    cryptonight::init_shared_aes_ttables(tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 1U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~1U;
    const unsigned int group = global_thread >> 1U;
    if (group >= count) return;
    const unsigned int mask = 0x3U << base;

    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + 256;
    const std::uint32_t* t2 = tables + 512;
    const std::uint32_t* t3 = tables + 768;

    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];
    std::uint64_t a = reinterpret_cast<const std::uint64_t*>(ctx.a)[lane];
    std::uint64_t b = reinterpret_cast<const std::uint64_t*>(ctx.b)[lane];
    const std::uint64_t tweak = ctx.tweak;

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint64_t a0 = __shfl_sync(mask, a, base);
        const std::uint32_t index = static_cast<std::uint32_t>((a0 >> 4U) & address_mask);
        std::uint8_t* slot = scratch + static_cast<std::size_t>(index) * 16U;

        const std::uint64_t s = reinterpret_cast<const std::uint64_t*>(slot)[lane];
        const std::uint64_t other = __shfl_sync(mask, s, base + (lane ^ 1U));
        const std::uint8_t* sb = reinterpret_cast<const std::uint8_t*>(&s);
        const std::uint8_t* ob = reinterpret_cast<const std::uint8_t*>(&other);

        const std::uint32_t cw_lo =
            t0[sb[0]] ^ t1[sb[5]] ^ t2[ob[2]] ^ t3[ob[7]] ^ static_cast<std::uint32_t>(a);
        const std::uint32_t cw_hi =
            t0[sb[4]] ^ t1[ob[1]] ^ t2[ob[6]] ^ t3[sb[3]] ^ static_cast<std::uint32_t>(a >> 32U);
        const std::uint64_t c = static_cast<std::uint64_t>(cw_lo) |
                                (static_cast<std::uint64_t>(cw_hi) << 32U);

        reinterpret_cast<std::uint64_t*>(slot)[lane] = c ^ b;
        __syncwarp(mask);
        if (lane == 0U) cryptonight::variant1_mutate(slot);
        __syncwarp(mask);

        const std::uint64_t c0 = __shfl_sync(mask, c, base);
        const std::uint32_t j = static_cast<std::uint32_t>((c0 >> 4U) & address_mask);
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(j) * 16U;
        const std::uint64_t tv = reinterpret_cast<const std::uint64_t*>(slot2)[lane];
        const std::uint64_t t0v = __shfl_sync(mask, tv, base);
        const std::uint64_t old_a1 = __shfl_sync(mask, a, base + 1U);

        std::uint64_t next0 = 0U;
        std::uint64_t next1 = 0U;
        if (lane == 0U) {
            const std::uint64_t lo = c0 * t0v;
            const std::uint64_t hi = __umul64hi(c0, t0v);
            next0 = a0 + hi;
            next1 = old_a1 + lo;
        }
        next0 = __shfl_sync(mask, next0, base);
        next1 = __shfl_sync(mask, next1, base);

        reinterpret_cast<std::uint64_t*>(slot2)[lane] = lane == 0U ? next0 : (next1 ^ tweak);
        a = (lane == 0U ? next0 : next1) ^ tv;
        b = c;
        __syncwarp(mask);
    }
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_coalesced(std::size_t count,
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
    const unsigned int pair_group = global_thread >> 2U;
    const std::size_t index_a = static_cast<std::size_t>(pair_group) * 2U;
    if (index_a >= count) return;
    const std::size_t index_b = index_a + 1U;
    const bool has_b = index_b < count;
    const unsigned int mask = 0xFU << base;

    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + 256;
    const std::uint32_t* t2 = tables + 512;
    const std::uint32_t* t3 = tables + 768;

    std::uint8_t* scratch_a = scratchpads + index_a * cfg.page_size;
    const auto& ctx_a = contexts[index_a];
    std::uint32_t aw_a = cryptonight::load32le_aligned(ctx_a.a + lane * 4U);
    std::uint32_t bw_a = cryptonight::load32le_aligned(ctx_a.b + lane * 4U);
    std::uint32_t aw1_shadow_a = 0U;
    if (lane == 0U) aw1_shadow_a = cryptonight::load32le_aligned(ctx_a.a + 4U);
    const std::uint32_t tweak_a_lo = static_cast<std::uint32_t>(ctx_a.tweak);
    const std::uint32_t tweak_a_hi = static_cast<std::uint32_t>(ctx_a.tweak >> 32U);

    std::uint8_t* scratch_b = nullptr;
    std::uint32_t aw_b = 0U;
    std::uint32_t bw_b = 0U;
    std::uint32_t aw1_shadow_b = 0U;
    std::uint32_t tweak_b_lo = 0U;
    std::uint32_t tweak_b_hi = 0U;
    if (has_b) {
        scratch_b = scratchpads + index_b * cfg.page_size;
        const auto& ctx_b = contexts[index_b];
        aw_b = cryptonight::load32le_aligned(ctx_b.a + lane * 4U);
        bw_b = cryptonight::load32le_aligned(ctx_b.b + lane * 4U);
        if (lane == 0U) aw1_shadow_b = cryptonight::load32le_aligned(ctx_b.a + 4U);
        tweak_b_lo = static_cast<std::uint32_t>(ctx_b.tweak);
        tweak_b_hi = static_cast<std::uint32_t>(ctx_b.tweak >> 32U);
    }

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint32_t aw0_a = __shfl_sync(mask, aw_a, base);
        const std::uint32_t aw1_a = lane == 0U ? aw1_shadow_a : 0U;
        std::uint8_t* slot_a = scratch_a + static_cast<std::size_t>(aw0_a & byte_address_mask);
        const std::uint32_t own_a = reinterpret_cast<const std::uint32_t*>(slot_a)[lane];

        std::uint32_t aw0_b = 0U;
        std::uint32_t aw1_b = 0U;
        std::uint8_t* slot_b = nullptr;
        std::uint32_t own_b = 0U;
        if (has_b) {
            aw0_b = __shfl_sync(mask, aw_b, base);
            aw1_b = lane == 0U ? aw1_shadow_b : 0U;
            slot_b = scratch_b + static_cast<std::size_t>(aw0_b & byte_address_mask);
            own_b = reinterpret_cast<const std::uint32_t*>(slot_b)[lane];
        }

        std::uint32_t cw_a = 0U;
        if constexpr (VariantIndex == 2) {
            const std::uint32_t q0 = t0[static_cast<std::uint8_t>(own_a)];
            const std::uint32_t q1_local = t1[static_cast<std::uint8_t>(own_a >> 8U)];
            const std::uint32_t q2_local = t2[static_cast<std::uint8_t>(own_a >> 16U)];
            const std::uint32_t q3_local = t3[static_cast<std::uint8_t>(own_a >> 24U)];
            const std::uint32_t q1 = __shfl_sync(mask, q1_local, base + ((lane + 1U) & 3U));
            const std::uint32_t q2 = __shfl_sync(mask, q2_local, base + ((lane + 2U) & 3U));
            const std::uint32_t q3 = __shfl_sync(mask, q3_local, base + ((lane + 3U) & 3U));
            cw_a = q0 ^ q1 ^ q2 ^ q3 ^ aw_a;
        } else {
            const std::uint32_t w1 = __shfl_sync(mask, own_a, base + ((lane + 1U) & 3U));
            const std::uint32_t w2 = __shfl_sync(mask, own_a, base + ((lane + 2U) & 3U));
            const std::uint32_t w3 = __shfl_sync(mask, own_a, base + ((lane + 3U) & 3U));
            cw_a = t0[static_cast<std::uint8_t>(own_a)] ^
                   t1[static_cast<std::uint8_t>(w1 >> 8U)] ^
                   t2[static_cast<std::uint8_t>(w2 >> 16U)] ^
                   t3[static_cast<std::uint8_t>(w3 >> 24U)] ^ aw_a;
        }

        std::uint32_t cw_b = 0U;
        if (has_b) {
            if constexpr (VariantIndex == 2) {
                const std::uint32_t q0 = t0[static_cast<std::uint8_t>(own_b)];
                const std::uint32_t q1_local = t1[static_cast<std::uint8_t>(own_b >> 8U)];
                const std::uint32_t q2_local = t2[static_cast<std::uint8_t>(own_b >> 16U)];
                const std::uint32_t q3_local = t3[static_cast<std::uint8_t>(own_b >> 24U)];
                const std::uint32_t q1 = __shfl_sync(mask, q1_local, base + ((lane + 1U) & 3U));
                const std::uint32_t q2 = __shfl_sync(mask, q2_local, base + ((lane + 2U) & 3U));
                const std::uint32_t q3 = __shfl_sync(mask, q3_local, base + ((lane + 3U) & 3U));
                cw_b = q0 ^ q1 ^ q2 ^ q3 ^ aw_b;
            } else {
                const std::uint32_t w1 = __shfl_sync(mask, own_b, base + ((lane + 1U) & 3U));
                const std::uint32_t w2 = __shfl_sync(mask, own_b, base + ((lane + 2U) & 3U));
                const std::uint32_t w3 = __shfl_sync(mask, own_b, base + ((lane + 3U) & 3U));
                cw_b = t0[static_cast<std::uint8_t>(own_b)] ^
                       t1[static_cast<std::uint8_t>(w1 >> 8U)] ^
                       t2[static_cast<std::uint8_t>(w2 >> 16U)] ^
                       t3[static_cast<std::uint8_t>(w3 >> 24U)] ^ aw_b;
            }
        }

        std::uint32_t first_a = cw_a ^ bw_a;
        if (lane == 2U) {
            const std::uint8_t tmp = static_cast<std::uint8_t>(first_a >> 24U);
            constexpr std::uint32_t table = 0x75310U;
            const std::uint8_t mutate_index =
                static_cast<std::uint8_t>((((tmp >> 3U) & 6U) | (tmp & 1U)) << 1U);
            first_a ^= ((table >> mutate_index) & 0x30U) << 24U;
        }
        reinterpret_cast<std::uint32_t*>(slot_a)[lane] = first_a;

        if (has_b) {
            std::uint32_t first_b = cw_b ^ bw_b;
            if (lane == 2U) {
                const std::uint8_t tmp = static_cast<std::uint8_t>(first_b >> 24U);
                constexpr std::uint32_t table = 0x75310U;
                const std::uint8_t mutate_index =
                    static_cast<std::uint8_t>((((tmp >> 3U) & 6U) | (tmp & 1U)) << 1U);
                first_b ^= ((table >> mutate_index) & 0x30U) << 24U;
            }
            reinterpret_cast<std::uint32_t*>(slot_b)[lane] = first_b;
        }

        const std::uint32_t c0_a = __shfl_sync(mask, cw_a, base);
        std::uint8_t* slot2_a = scratch_a + static_cast<std::size_t>(c0_a & byte_address_mask);
        const std::uint32_t t_word_a = reinterpret_cast<const std::uint32_t*>(slot2_a)[lane];

        std::uint32_t c0_b = 0U;
        std::uint8_t* slot2_b = nullptr;
        std::uint32_t t_word_b = 0U;
        if (has_b) {
            c0_b = __shfl_sync(mask, cw_b, base);
            slot2_b = scratch_b + static_cast<std::size_t>(c0_b & byte_address_mask);
            t_word_b = reinterpret_cast<const std::uint32_t*>(slot2_b)[lane];
        }

        const std::uint32_t c1_a = __shfl_sync(mask, cw_a, base + 1U);
        const std::uint32_t t0_hi_a = __shfl_sync(mask, t_word_a, base + 1U);
        const std::uint32_t old_a2_a = __shfl_sync(mask, aw_a, base + 2U);
        const std::uint32_t old_a3_a = __shfl_sync(mask, aw_a, base + 3U);
        const std::uint32_t t0_lo_a = lane == 0U ? t_word_a : 0U;

        std::uint32_t c1_b = 0U, t0_hi_b = 0U, old_a2_b = 0U, old_a3_b = 0U, t0_lo_b = 0U;
        if (has_b) {
            c1_b = __shfl_sync(mask, cw_b, base + 1U);
            t0_hi_b = __shfl_sync(mask, t_word_b, base + 1U);
            old_a2_b = __shfl_sync(mask, aw_b, base + 2U);
            old_a3_b = __shfl_sync(mask, aw_b, base + 3U);
            t0_lo_b = lane == 0U ? t_word_b : 0U;
        }

        std::uint32_t na0_lo_a = 0U, na0_hi_a = 0U, na1_lo_a = 0U, na1_hi_a = 0U;
        std::uint32_t na0_lo_b = 0U, na0_hi_b = 0U, na1_lo_b = 0U, na1_hi_b = 0U;
        if (lane == 0U) {
            const std::uint64_t c_low_a = static_cast<std::uint64_t>(c0_a) |
                                          (static_cast<std::uint64_t>(c1_a) << 32U);
            const std::uint64_t t0v_a = static_cast<std::uint64_t>(t0_lo_a) |
                                        (static_cast<std::uint64_t>(t0_hi_a) << 32U);
            const std::uint64_t lo_a = c_low_a * t0v_a;
            const std::uint64_t hi_a = __umul64hi(c_low_a, t0v_a);

            std::uint64_t lo_b = 0U;
            std::uint64_t hi_b = 0U;
            std::uint64_t c_low_b = 0U;
            std::uint64_t t0v_b = 0U;
            if (has_b) {
                c_low_b = static_cast<std::uint64_t>(c0_b) |
                          (static_cast<std::uint64_t>(c1_b) << 32U);
                t0v_b = static_cast<std::uint64_t>(t0_lo_b) |
                        (static_cast<std::uint64_t>(t0_hi_b) << 32U);
                lo_b = c_low_b * t0v_b;
                hi_b = __umul64hi(c_low_b, t0v_b);
            }

            const std::uint64_t olda0_a = static_cast<std::uint64_t>(aw0_a) |
                                          (static_cast<std::uint64_t>(aw1_a) << 32U);
            const std::uint64_t olda1_a = static_cast<std::uint64_t>(old_a2_a) |
                                          (static_cast<std::uint64_t>(old_a3_a) << 32U);
            const std::uint64_t na0_a = olda0_a + hi_a;
            const std::uint64_t na1_a = olda1_a + lo_a;
            na0_lo_a = static_cast<std::uint32_t>(na0_a);
            na0_hi_a = static_cast<std::uint32_t>(na0_a >> 32U);
            na1_lo_a = static_cast<std::uint32_t>(na1_a);
            na1_hi_a = static_cast<std::uint32_t>(na1_a >> 32U);
            aw1_shadow_a = na0_hi_a ^ t0_hi_a;

            if (has_b) {
                const std::uint64_t olda0_b = static_cast<std::uint64_t>(aw0_b) |
                                              (static_cast<std::uint64_t>(aw1_b) << 32U);
                const std::uint64_t olda1_b = static_cast<std::uint64_t>(old_a2_b) |
                                              (static_cast<std::uint64_t>(old_a3_b) << 32U);
                const std::uint64_t na0_b = olda0_b + hi_b;
                const std::uint64_t na1_b = olda1_b + lo_b;
                na0_lo_b = static_cast<std::uint32_t>(na0_b);
                na0_hi_b = static_cast<std::uint32_t>(na0_b >> 32U);
                na1_lo_b = static_cast<std::uint32_t>(na1_b);
                na1_hi_b = static_cast<std::uint32_t>(na1_b >> 32U);
                aw1_shadow_b = na0_hi_b ^ t0_hi_b;
            }
        }

        na0_lo_a = __shfl_sync(mask, na0_lo_a, base);
        na0_hi_a = __shfl_sync(mask, na0_hi_a, base);
        na1_lo_a = __shfl_sync(mask, na1_lo_a, base);
        na1_hi_a = __shfl_sync(mask, na1_hi_a, base);

        if (has_b) {
            na0_lo_b = __shfl_sync(mask, na0_lo_b, base);
            na0_hi_b = __shfl_sync(mask, na0_hi_b, base);
            na1_lo_b = __shfl_sync(mask, na1_lo_b, base);
            na1_hi_b = __shfl_sync(mask, na1_hi_b, base);
        }

        const std::uint32_t store_a = lane == 0U ? na0_lo_a :
                                      lane == 1U ? na0_hi_a :
                                      lane == 2U ? (na1_lo_a ^ tweak_a_lo) : (na1_hi_a ^ tweak_a_hi);
        reinterpret_cast<std::uint32_t*>(slot2_a)[lane] = store_a;
        const std::uint32_t na_word_a = lane == 0U ? na0_lo_a :
                                        lane == 1U ? na0_hi_a :
                                        lane == 2U ? na1_lo_a : na1_hi_a;
        aw_a = na_word_a ^ t_word_a;
        bw_a = cw_a;

        if (has_b) {
            const std::uint32_t store_b = lane == 0U ? na0_lo_b :
                                          lane == 1U ? na0_hi_b :
                                          lane == 2U ? (na1_lo_b ^ tweak_b_lo) : (na1_hi_b ^ tweak_b_hi);
            reinterpret_cast<std::uint32_t*>(slot2_b)[lane] = store_b;
            const std::uint32_t na_word_b = lane == 0U ? na0_lo_b :
                                            lane == 1U ? na0_hi_b :
                                            lane == 2U ? na1_lo_b : na1_hi_b;
            aw_b = na_word_b ^ t_word_b;
            bw_b = cw_b;
        }
    }
}

} // anonymous namespace
