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

    // The dependent scratchpad addresses use only bits 4..N from the low
    // 32-bit word for every GhostRider CN variant. Lane 0 is also the only lane
    // that consumes olda0's high word during the multiply/add step, so carry a
    // local shadow instead of rediscovering lane 1's A word every iteration.
    std::uint32_t aw1_shadow = 0U;
    if (lane == 0U) aw1_shadow = cryptonight::load32le_aligned(ctx.a + 4U);

    const std::uint64_t tweak = ctx.tweak;
    const std::uint32_t tweak_lo = static_cast<std::uint32_t>(tweak);
    const std::uint32_t tweak_hi = static_cast<std::uint32_t>(tweak >> 32U);

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        // ((a_low >> 4) & address_mask) * 16 is exactly the low word masked
        // with (address_mask << 4). This removes a subgroup shuffle, 64-bit
        // combine and widened multiply from the first dependent address path.
        const std::uint32_t aw0 = __shfl_sync(mask, aw, base);
        const std::uint32_t aw1 = lane == 0U ? aw1_shadow : 0U;
        const std::uint32_t byte_offset = aw0 & byte_address_mask;
        std::uint8_t* slot = scratch + static_cast<std::size_t>(byte_offset);

        const std::uint32_t own = reinterpret_cast<const std::uint32_t*>(slot)[lane];
        std::uint32_t cw = 0U;
        if constexpr (VariantIndex == 2) {
            // CN-Fast benefits from issuing all four local shared-table reads
            // before the ShiftRows exchange. Other variants keep the established
            // source-word shuffle ordering unless a separate candidate changes it.
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

        // The second dependent address obeys the same low-word identity. Issue
        // the memory access before fetching c1/remaining A words so shuffle
        // latency is kept off the address dependency chain.
        const std::uint32_t c0 = __shfl_sync(mask, cw, base);
        const std::uint32_t byte_offset2 = c0 & byte_address_mask;
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(byte_offset2);

        const std::uint32_t t_word = reinterpret_cast<const std::uint32_t*>(slot2)[lane];
        const std::uint32_t t0_lo = lane == 0U ? t_word : 0U;
        const std::uint32_t c1 = __shfl_sync(mask, cw, base + 1U);
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t na0_lo = 0U, na0_hi = 0U, na1_lo = 0U, na1_hi = 0U;
        if (lane == 0U) {
            const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                        (static_cast<std::uint64_t>(c1) << 32U);
            const std::uint64_t t0v =
                static_cast<std::uint64_t>(t0_lo) |
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

            // Next iteration's lane-1 A word is exactly na0_hi ^ t0_hi.
            aw1_shadow = na0_hi ^ t0_hi;
        }

        na0_lo = __shfl_sync(mask, na0_lo, base);
        na0_hi = __shfl_sync(mask, na0_hi, base);
        na1_lo = __shfl_sync(mask, na1_lo, base);
        na1_hi = __shfl_sync(mask, na1_hi, base);
        const std::uint32_t na_word = lane == 0U ? na0_lo :
                                      lane == 1U ? na0_hi :
                                      lane == 2U ? na1_lo : na1_hi;

        std::uint32_t store_word = na_word;
        if (lane == 2U) store_word ^= tweak_lo;
        if (lane == 3U) store_word ^= tweak_hi;
        reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

        aw = na_word ^ t_word;
        bw = cw;
    }
}


// Experimental mode 445: two independent hashes per 4-lane subgroup.
//
// CryptoNight phase 2 is a long serial dependency chain. More resident warps,
// lower register count, different block sizes and small shuffle rearrangements
// did not materially improve throughput. This kernel instead creates explicit
// instruction-level parallelism inside each subgroup: it carries two unrelated
// hash states and issues the dependent scratchpad operations for both hashes
// before consuming either result.
//
// The arithmetic and memory semantics for each hash remain identical to the
// proven four-lane T-table baseline. Only scheduling/mapping changes.
template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_dualhash(
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
    const unsigned int pair_group = global_thread >> 2U;
    const std::size_t hash_a = static_cast<std::size_t>(pair_group) * 2U;
    if (hash_a >= count) return;
    const std::size_t hash_b = hash_a + 1U;
    const bool has_b = hash_b < count;
    const unsigned int mask = 0xFU << base;

    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + 256;
    const std::uint32_t* t2 = tables + 512;
    const std::uint32_t* t3 = tables + 768;

    std::uint8_t* scratch_a =
        scratchpads + hash_a * cfg.page_size;
    std::uint8_t* scratch_b =
        has_b ? scratchpads + hash_b * cfg.page_size : nullptr;

    const auto& ctx_a = contexts[hash_a];
    std::uint32_t aw_a = cryptonight::load32le_aligned(ctx_a.a + lane * 4U);
    std::uint32_t bw_a = cryptonight::load32le_aligned(ctx_a.b + lane * 4U);
    std::uint32_t aw1_a = 0U;
    if (lane == 0U) aw1_a = cryptonight::load32le_aligned(ctx_a.a + 4U);
    const std::uint64_t tweak_a = ctx_a.tweak;
    const std::uint32_t tweak_a_lo = static_cast<std::uint32_t>(tweak_a);
    const std::uint32_t tweak_a_hi = static_cast<std::uint32_t>(tweak_a >> 32U);

    std::uint32_t aw_b = 0U;
    std::uint32_t bw_b = 0U;
    std::uint32_t aw1_b = 0U;
    std::uint32_t tweak_b_lo = 0U;
    std::uint32_t tweak_b_hi = 0U;
    if (has_b) {
        const auto& ctx_b = contexts[hash_b];
        aw_b = cryptonight::load32le_aligned(ctx_b.a + lane * 4U);
        bw_b = cryptonight::load32le_aligned(ctx_b.b + lane * 4U);
        if (lane == 0U) aw1_b = cryptonight::load32le_aligned(ctx_b.a + 4U);
        const std::uint64_t tweak_b = ctx_b.tweak;
        tweak_b_lo = static_cast<std::uint32_t>(tweak_b);
        tweak_b_hi = static_cast<std::uint32_t>(tweak_b >> 32U);
    }

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        // First dependent address for A and B. Both global loads are issued
        // before either result enters the AES T-table dependency path.
        const std::uint32_t aw0_a = __shfl_sync(mask, aw_a, base);
        const std::uint32_t off_a = aw0_a & byte_address_mask;
        std::uint8_t* slot_a = scratch_a + static_cast<std::size_t>(off_a);
        const std::uint32_t own_a =
            reinterpret_cast<const std::uint32_t*>(slot_a)[lane];

        std::uint32_t aw0_b = 0U;
        std::uint8_t* slot_b = nullptr;
        std::uint32_t own_b = 0U;
        if (has_b) {
            aw0_b = __shfl_sync(mask, aw_b, base);
            const std::uint32_t off_b = aw0_b & byte_address_mask;
            slot_b = scratch_b + static_cast<std::size_t>(off_b);
            own_b = reinterpret_cast<const std::uint32_t*>(slot_b)[lane];
        }

        auto aes_word = [&](std::uint32_t own, std::uint32_t aw) {
            if constexpr (VariantIndex == 2) {
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
                return q0 ^ q1 ^ q2 ^ q3 ^ aw;
            } else {
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
                return t0[x0] ^ t1[x1] ^ t2[x2] ^ t3[x3] ^ aw;
            }
        };

        const std::uint32_t cw_a = aes_word(own_a, aw_a);
        std::uint32_t cw_b = 0U;
        if (has_b) cw_b = aes_word(own_b, aw_b);

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

        // Second dependent address/load for both hashes before either multiply.
        const std::uint32_t c0_a = __shfl_sync(mask, cw_a, base);
        std::uint8_t* slot2_a =
            scratch_a + static_cast<std::size_t>(c0_a & byte_address_mask);
        const std::uint32_t tv_a =
            reinterpret_cast<const std::uint32_t*>(slot2_a)[lane];

        std::uint32_t c0_b = 0U;
        std::uint8_t* slot2_b = nullptr;
        std::uint32_t tv_b = 0U;
        if (has_b) {
            c0_b = __shfl_sync(mask, cw_b, base);
            slot2_b =
                scratch_b + static_cast<std::size_t>(c0_b & byte_address_mask);
            tv_b = reinterpret_cast<const std::uint32_t*>(slot2_b)[lane];
        }

        auto finish = [&](std::uint32_t aw,
                          std::uint32_t aw0,
                          std::uint32_t aw1,
                          std::uint32_t cw,
                          std::uint32_t c0,
                          std::uint32_t tv,
                          std::uint32_t tweak_lo,
                          std::uint32_t tweak_hi,
                          std::uint8_t* slot2,
                          std::uint32_t& next_aw1,
                          std::uint32_t& next_aw,
                          std::uint32_t& next_bw) {
            const std::uint32_t c1 = __shfl_sync(mask, cw, base + 1U);
            const std::uint32_t t0_hi = __shfl_sync(mask, tv, base + 1U);
            const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
            const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

            std::uint32_t na0_lo = 0U, na0_hi = 0U, na1_lo = 0U, na1_hi = 0U;
            if (lane == 0U) {
                const std::uint64_t c_low =
                    static_cast<std::uint64_t>(c0) |
                    (static_cast<std::uint64_t>(c1) << 32U);
                const std::uint64_t t0v =
                    static_cast<std::uint64_t>(tv) |
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
                next_aw1 = na0_hi ^ t0_hi;
            }

            na0_lo = __shfl_sync(mask, na0_lo, base);
            na0_hi = __shfl_sync(mask, na0_hi, base);
            na1_lo = __shfl_sync(mask, na1_lo, base);
            na1_hi = __shfl_sync(mask, na1_hi, base);
            const std::uint32_t na_word =
                lane == 0U ? na0_lo :
                lane == 1U ? na0_hi :
                lane == 2U ? na1_lo : na1_hi;

            std::uint32_t store_word = na_word;
            if (lane == 2U) store_word ^= tweak_lo;
            if (lane == 3U) store_word ^= tweak_hi;
            reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

            next_aw = na_word ^ tv;
            next_bw = cw;
        };

        finish(aw_a, aw0_a, aw1_a, cw_a, c0_a, tv_a,
               tweak_a_lo, tweak_a_hi, slot2_a, aw1_a, aw_a, bw_a);

        if (has_b) {
            finish(aw_b, aw0_b, aw1_b, cw_b, c0_b, tv_b,
                   tweak_b_lo, tweak_b_hi, slot2_b, aw1_b, aw_b, bw_b);
        }
    }
}

} // anonymous namespace