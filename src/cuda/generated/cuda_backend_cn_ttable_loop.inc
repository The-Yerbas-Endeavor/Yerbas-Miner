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
// Neither path is trusted by construction: the production selector validates
// finalized output against cryptonight_loop_stage_cooperative before promotion.

namespace {

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable2(std::size_t count,
                                               std::uint8_t* scratchpads,
                                               const cryptonight::SplitContext* contexts)
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
                                                         std::uint8_t* scratchpads,
                                                         const cryptonight::SplitContext* contexts)
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

    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];
    std::uint32_t aw = cryptonight::load32le_aligned(ctx.a + lane * 4U);
    std::uint32_t bw = cryptonight::load32le_aligned(ctx.b + lane * 4U);
    std::uint32_t fast_aw1_shadow = 0U;
    if constexpr (VariantIndex == 2) {
        // Lane 0 is the only lane that consumes the high word of olda0 during
        // the multiply/add dependency step. Keep a lane-0 shadow so CN-Fast
        // does not have to fetch the same word from lane 1 every iteration.
        if (lane == 0U) fast_aw1_shadow = cryptonight::load32le_aligned(ctx.a + 4U);
    }
    const std::uint64_t tweak = ctx.tweak;
    const std::uint32_t tweak_lo = static_cast<std::uint32_t>(tweak);
    const std::uint32_t tweak_hi = static_cast<std::uint32_t>(tweak >> 32U);

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        std::uint32_t aw0 = 0U;
        std::uint32_t aw1 = 0U;
        std::uint8_t* slot = nullptr;

        if constexpr (VariantIndex == 2) {
            // CN-Fast's index expression ((aw0 >> 4) & address_mask) * 16
            // is exactly aw0 & (address_mask << 4). Keep the dependency path
            // in 32 bits and issue the scratchpad access from one mask instead
            // of shift + mask + widened multiply-by-16 address arithmetic.
            aw0 = __shfl_sync(mask, aw, base);
            const std::uint32_t byte_offset = aw0 & byte_address_mask;
            slot = scratch + static_cast<std::size_t>(byte_offset);
            if (lane == 0U) aw1 = fast_aw1_shadow;
        } else {
            aw0 = __shfl_sync(mask, aw, base);
            aw1 = __shfl_sync(mask, aw, base + 1U);
            const std::uint64_t a_low = static_cast<std::uint64_t>(aw0) |
                                        (static_cast<std::uint64_t>(aw1) << 32U);
            const std::uint32_t index = static_cast<std::uint32_t>((a_low >> 4U) & address_mask);
            slot = scratch + static_cast<std::size_t>(index) * 16U;
        }

        const std::uint32_t own = reinterpret_cast<const std::uint32_t*>(slot)[lane];
        std::uint32_t cw = 0U;
        if constexpr (VariantIndex == 2) {
            // Each lane owns one 32-bit AES input word. Perform all four shared
            // T-table reads from that local word first, before any ShiftRows
            // exchange. This exposes the independent table loads to the GPU
            // together instead of serializing them behind three source-word
            // shuffles. Then route the already-computed table contributions to
            // the destination lane that consumes each ShiftRows byte.
            const std::uint32_t q0 = t0[static_cast<std::uint8_t>(own)];
            const std::uint32_t q1_local = t1[static_cast<std::uint8_t>(own >> 8U)];
            const std::uint32_t q2_local = t2[static_cast<std::uint8_t>(own >> 16U)];
            const std::uint32_t q3_local = t3[static_cast<std::uint8_t>(own >> 24U)];
            const std::uint32_t q1 = __shfl_sync(mask, q1_local, base + ((lane + 1U) & 3U));
            const std::uint32_t q2 = __shfl_sync(mask, q2_local, base + ((lane + 2U) & 3U));
            const std::uint32_t q3 = __shfl_sync(mask, q3_local, base + ((lane + 3U) & 3U));
            cw = q0 ^ q1 ^ q2 ^ q3 ^ aw;
        } else {
            const std::uint32_t w1 = __shfl_sync(mask, own, base + ((lane + 1U) & 3U));
            const std::uint32_t w2 = __shfl_sync(mask, own, base + ((lane + 2U) & 3U));
            const std::uint32_t w3 = __shfl_sync(mask, own, base + ((lane + 3U) & 3U));
            const std::uint8_t x0 = static_cast<std::uint8_t>(own);
            const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
            const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
            const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);
            cw = t0[x0] ^ t1[x1] ^ t2[x2] ^ t3[x3] ^ aw;
        }

        std::uint32_t first_store_word = cw ^ bw;
        if (lane == 2U) {
            const std::uint8_t tmp = static_cast<std::uint8_t>(first_store_word >> 24U);
            constexpr std::uint32_t table = 0x75310U;
            const std::uint8_t mutate_index =
                static_cast<std::uint8_t>((((tmp >> 3U) & 6U) | (tmp & 1U)) << 1U);
            first_store_word ^= ((table >> mutate_index) & 0x30U) << 24U;
        }
        reinterpret_cast<std::uint32_t*>(slot)[lane] = first_store_word;

        std::uint32_t c0 = 0U;
        std::uint32_t c1 = 0U;
        std::uint8_t* slot2 = nullptr;
        if constexpr (VariantIndex == 2) {
            // Same strength reduction for the second dependent access. c1 and
            // the remaining A words stay off the address path until slot2 has
            // already been issued.
            c0 = __shfl_sync(mask, cw, base);
            const std::uint32_t byte_offset = c0 & byte_address_mask;
            slot2 = scratch + static_cast<std::size_t>(byte_offset);
        } else {
            c0 = __shfl_sync(mask, cw, base);
            c1 = __shfl_sync(mask, cw, base + 1U);
            const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                        (static_cast<std::uint64_t>(c1) << 32U);
            const std::uint32_t j = static_cast<std::uint32_t>((c_low >> 4U) & address_mask);
            slot2 = scratch + static_cast<std::size_t>(j) * 16U;
        }

        const std::uint32_t t_word = reinterpret_cast<const std::uint32_t*>(slot2)[lane];
        std::uint32_t t0_lo = 0U;
        if constexpr (VariantIndex == 2) {
            t0_lo = lane == 0U ? t_word : 0U;
            // c1 is not part of either scratchpad address. Fetch it only after
            // slot2 has been issued so its shuffle latency stays off the memory
            // address path. aw1 comes from lane 0's carried shadow instead.
            c1 = __shfl_sync(mask, cw, base + 1U);
        } else {
            t0_lo = __shfl_sync(mask, t_word, base);
        }
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t na0_lo = 0U, na0_hi = 0U, na1_lo = 0U, na1_hi = 0U;
        if (lane == 0U) {
            const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                        (static_cast<std::uint64_t>(c1) << 32U);
            const std::uint64_t t0v = static_cast<std::uint64_t>(t0_lo) |
                                      (static_cast<std::uint64_t>(t0_hi) << 32U);
            const std::uint64_t lo = c_low * t0v;
            const std::uint64_t hi = __umul64hi(c_low, t0v);
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
            if constexpr (VariantIndex == 2) {
                // Next iteration's lane-1 A word is exactly na0_hi ^ t0_hi.
                // Carry it here instead of rediscovering it with a subgroup
                // shuffle from lane 1 on the following iteration.
                fast_aw1_shadow = na0_hi ^ t0_hi;
            }
        }
        na0_lo = __shfl_sync(mask, na0_lo, base);
        na0_hi = __shfl_sync(mask, na0_hi, base);
        na1_lo = __shfl_sync(mask, na1_lo, base);
        na1_hi = __shfl_sync(mask, na1_hi, base);

        const std::uint32_t store_word = lane == 0U ? na0_lo :
                                         lane == 1U ? na0_hi :
                                         lane == 2U ? (na1_lo ^ tweak_lo) : (na1_hi ^ tweak_hi);
        reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

        const std::uint32_t na_word = lane == 0U ? na0_lo :
                                      lane == 1U ? na0_hi :
                                      lane == 2U ? na1_lo : na1_hi;
        aw = na_word ^ t_word;
        bw = cw;
    }
}

} // anonymous namespace