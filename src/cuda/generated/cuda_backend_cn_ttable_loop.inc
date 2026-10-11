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

__device__ __align__(128) std::uint32_t g_cn_aes_ttables_readonly[1024];

__global__ void init_cn_aes_ttables_readonly()
{
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 256) return;
    const std::uint8_t s = cryptonight::kAesSbox[i];
    const std::uint8_t x2 = cryptonight::xtime(s);
    const std::uint8_t x3 = static_cast<std::uint8_t>(x2 ^ s);
    g_cn_aes_ttables_readonly[i]       = cryptonight::pack4(x2, s,  s,  x3);
    g_cn_aes_ttables_readonly[256 + i] = cryptonight::pack4(x3, x2, s,  s );
    g_cn_aes_ttables_readonly[512 + i] = cryptonight::pack4(s,  x3, x2, s );
    g_cn_aes_ttables_readonly[768 + i] = cryptonight::pack4(s,  s,  x3, x2);
}

__device__ __forceinline__ std::uint32_t cn_aes_readonly_load(
    const std::uint32_t* base, std::uint32_t index)
{
    return __ldg(base + index);
}

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


__device__ __forceinline__ void cn_init_shared_aes_ttables_bank8(
    std::uint32_t* tables)
{
    // Layout per table: [256 logical indices][8 duplicated lane classes].
    // Pascal shared-memory banks are four bytes wide. The lane class in the
    // low address bits reduces unrelated random lookups from collapsing onto
    // the same bank as often as the ordinary 1x T-table layout.
    constexpr int kReplicas = 8;
    constexpr int kTableWords = 256 * kReplicas;
    constexpr int kTotalWords = 4 * kTableWords;

    for (int logical = static_cast<int>(threadIdx.x);
         logical < kTotalWords;
         logical += static_cast<int>(blockDim.x)) {
        const int table = logical / kTableWords;
        const int rem = logical - table * kTableWords;
        const int index = rem / kReplicas;

        const std::uint8_t s = cryptonight::kAesSbox[index];
        const std::uint8_t x2 = cryptonight::xtime(s);
        const std::uint8_t x3 = static_cast<std::uint8_t>(x2 ^ s);

        std::uint32_t value = 0U;
        if (table == 0) {
            value = cryptonight::pack4(x2, s,  s,  x3);
        } else if (table == 1) {
            value = cryptonight::pack4(x3, x2, s,  s );
        } else if (table == 2) {
            value = cryptonight::pack4(s,  x3, x2, s );
        } else {
            value = cryptonight::pack4(s,  s,  x3, x2);
        }
        tables[logical] = value;
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

// Sparse-warp phase-2 experiment.
//
// The production kernel packs eight independent four-lane hashes into each
// 32-lane warp. Random scratchpad latency is therefore paid at warp granularity:
// one slow subgroup can keep seven unrelated hashes from issuing. These
// specializations deliberately leave lanes idle so independent hashes occupy
// more independently schedulable warps without changing the per-hash state
// machine or scratchpad traffic.
//
// HashesPerWarp=4 -> 16 active lanes / warp, 896 warps at batch 3584.
// HashesPerWarp=2 ->  8 active lanes / warp, 1792 warps at batch 3584.

// Eight-way lane-classed shared AES T-table experiment.
//
// The baseline stores one 4 KiB T-table set. With random indices, all 32 lanes
// in a warp can collide on the same 32 Pascal shared-memory banks. This layout
// expands the tables to 32 KiB and stores eight duplicate lane classes per
// logical entry. The low address bits now encode warp_lane&7, reducing
// unrelated cross-lane bank collisions without changing any AES value,
// scratchpad address, or CryptoNight state transition.
//
template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_bank8(std::size_t count,
                                                         std::uint8_t* __restrict__ scratchpads,
                                                         const cryptonight::SplitContext* __restrict__ contexts)
{
    __shared__ std::uint32_t tables[8192];
    cn_init_shared_aes_ttables_bank8(tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    constexpr std::uint32_t byte_address_mask =
        static_cast<std::uint32_t>(address_mask << 4U);
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~3U;
    const unsigned int table_lane = warp_lane & 7U;
    const unsigned int group = global_thread >> 2U;
    if (group >= count) return;
    const unsigned int mask = 0xFU << base;

    constexpr unsigned int kReplicas = 8U;
    constexpr unsigned int kTableWords = 256U * kReplicas;
    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + kTableWords;
    const std::uint32_t* t2 = tables + 2U * kTableWords;
    const std::uint32_t* t3 = tables + 3U * kTableWords;

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
            const std::uint32_t q0 = t0[static_cast<unsigned int>(static_cast<std::uint8_t>(own)) * kReplicas + table_lane];
            const std::uint32_t q1_local = t1[static_cast<unsigned int>(static_cast<std::uint8_t>(own >> 8U)) * kReplicas + table_lane];
            const std::uint32_t q2_local = t2[static_cast<unsigned int>(static_cast<std::uint8_t>(own >> 16U)) * kReplicas + table_lane];
            const std::uint32_t q3_local = t3[static_cast<unsigned int>(static_cast<std::uint8_t>(own >> 24U)) * kReplicas + table_lane];
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
            cw = t0[static_cast<unsigned int>(x0) * kReplicas + table_lane] ^ t1[static_cast<unsigned int>(x1) * kReplicas + table_lane] ^ t2[static_cast<unsigned int>(x2) * kReplicas + table_lane] ^ t3[static_cast<unsigned int>(x3) * kReplicas + table_lane] ^ aw;
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

// Sparse-warp phase-2 experiment.
//
// The production kernel packs eight independent four-lane hashes into each
// 32-lane warp. Random scratchpad latency is therefore paid at warp granularity:
// one slow subgroup can keep seven unrelated hashes from issuing. These
// specializations deliberately leave lanes idle so independent hashes occupy
// more independently schedulable warps without changing the per-hash state
// machine or scratchpad traffic.
//
// HashesPerWarp=4 -> 16 active lanes / warp, 896 warps at batch 3584.
// HashesPerWarp=2 ->  8 active lanes / warp, 1792 warps at batch 3584.
template <std::uint8_t VariantIndex, unsigned int HashesPerWarp>
__global__ void cryptonight_loop_stage_ttable4_sparsewarp(std::size_t count,
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
    static_assert(HashesPerWarp == 2U || HashesPerWarp == 4U,
                  "sparse-warp experiment supports 2 or 4 hashes per warp");
    const unsigned int global_thread =
        blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int subgroup = warp_lane >> 2U;
    const unsigned int lane = warp_lane & 3U;
    const unsigned int base = warp_lane & ~3U;
    const std::size_t global_warp =
        static_cast<std::size_t>(global_thread >> 5U);
    const std::size_t group =
        global_warp * HashesPerWarp + subgroup;
    if (subgroup >= HashesPerWarp || group >= count) return;
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


template <std::uint8_t VariantIndex>
__device__ __forceinline__ std::uint32_t cn_ttable4_aes_word(
    std::uint32_t own,
    std::uint32_t aw,
    const std::uint32_t* t0,
    const std::uint32_t* t1,
    const std::uint32_t* t2,
    const std::uint32_t* t3,
    unsigned int lane,
    unsigned int base,
    unsigned int mask)
{
    if constexpr (VariantIndex == 2) {
        const std::uint32_t q0 = t0[static_cast<std::uint8_t>(own)];
        const std::uint32_t q1_local = t1[static_cast<std::uint8_t>(own >> 8U)];
        const std::uint32_t q2_local = t2[static_cast<std::uint8_t>(own >> 16U)];
        const std::uint32_t q3_local = t3[static_cast<std::uint8_t>(own >> 24U)];
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
}

__device__ __forceinline__ std::uint32_t cn_ttable4_first_store_word(
    std::uint32_t cw,
    std::uint32_t bw,
    unsigned int lane)
{
    std::uint32_t word = cw ^ bw;
    if (lane == 2U) {
        const std::uint8_t tmp = static_cast<std::uint8_t>(word >> 24U);
        constexpr std::uint32_t table = 0x75310U;
        const std::uint8_t mutate_index =
            static_cast<std::uint8_t>((((tmp >> 3U) & 6U) |
                                       (tmp & 1U)) << 1U);
        word ^= ((table >> mutate_index) & 0x30U) << 24U;
    }
    return word;
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_dualhash(std::size_t count,
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
    const unsigned int pair = global_thread >> 2U;
    const std::size_t group_a = static_cast<std::size_t>(pair) * 2U;
    if (group_a >= count) return;
    const std::size_t group_b = group_a + 1U;
    const bool have_b = group_b < count;
    const unsigned int mask = 0xFU << base;

    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + 256;
    const std::uint32_t* t2 = tables + 512;
    const std::uint32_t* t3 = tables + 768;

    std::uint8_t* scratch_a =
        scratchpads + group_a * cfg.page_size;
    std::uint8_t* scratch_b =
        scratchpads + (have_b ? group_b : group_a) * cfg.page_size;

    const auto& ctx_a = contexts[group_a];
    const auto& ctx_b = contexts[have_b ? group_b : group_a];

    std::uint32_t aw_a = cryptonight::load32le_aligned(ctx_a.a + lane * 4U);
    std::uint32_t bw_a = cryptonight::load32le_aligned(ctx_a.b + lane * 4U);
    std::uint32_t aw_b = cryptonight::load32le_aligned(ctx_b.a + lane * 4U);
    std::uint32_t bw_b = cryptonight::load32le_aligned(ctx_b.b + lane * 4U);

    std::uint32_t aw1_shadow_a = 0U;
    std::uint32_t aw1_shadow_b = 0U;
    if (lane == 0U) {
        aw1_shadow_a = cryptonight::load32le_aligned(ctx_a.a + 4U);
        aw1_shadow_b = cryptonight::load32le_aligned(ctx_b.a + 4U);
    }

    const std::uint64_t tweak_a = ctx_a.tweak;
    const std::uint64_t tweak_b = ctx_b.tweak;
    const std::uint32_t tweak_lo_a = static_cast<std::uint32_t>(tweak_a);
    const std::uint32_t tweak_hi_a = static_cast<std::uint32_t>(tweak_a >> 32U);
    const std::uint32_t tweak_lo_b = static_cast<std::uint32_t>(tweak_b);
    const std::uint32_t tweak_hi_b = static_cast<std::uint32_t>(tweak_b >> 32U);

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint32_t aw0_a = __shfl_sync(mask, aw_a, base);
        const std::uint32_t aw0_b = __shfl_sync(mask, aw_b, base);
        const std::uint32_t aw1_a = lane == 0U ? aw1_shadow_a : 0U;
        const std::uint32_t aw1_b = lane == 0U ? aw1_shadow_b : 0U;

        std::uint8_t* slot_a =
            scratch_a + static_cast<std::size_t>(aw0_a & byte_address_mask);
        std::uint8_t* slot_b =
            scratch_b + static_cast<std::size_t>(aw0_b & byte_address_mask);

        // True per-lane ILP: issue both independent random scratchpad reads
        // before consuming either result.  Unlike the retired split-warp
        // experiment, every participating lane carries state for two hashes.
        const auto* own_addr_a =
            reinterpret_cast<const std::uint32_t*>(slot_a) + lane;
        const auto* own_addr_b =
            reinterpret_cast<const std::uint32_t*>(slot_b) + lane;
        std::uint32_t own_a = 0U;
        std::uint32_t own_b = 0U;
        asm volatile(
            "ld.global.u32 %0, [%2];\n\t"
            "ld.global.u32 %1, [%3];\n\t"
            : "=r"(own_a), "=r"(own_b)
            : "l"(own_addr_a), "l"(own_addr_b));

        const std::uint32_t cw_a =
            cn_ttable4_aes_word<VariantIndex>(
                own_a, aw_a, t0, t1, t2, t3, lane, base, mask);
        const std::uint32_t cw_b =
            cn_ttable4_aes_word<VariantIndex>(
                own_b, aw_b, t0, t1, t2, t3, lane, base, mask);

        reinterpret_cast<std::uint32_t*>(slot_a)[lane] =
            cn_ttable4_first_store_word(cw_a, bw_a, lane);
        if (have_b) {
            reinterpret_cast<std::uint32_t*>(slot_b)[lane] =
                cn_ttable4_first_store_word(cw_b, bw_b, lane);
        }

        const std::uint32_t c0_a = __shfl_sync(mask, cw_a, base);
        const std::uint32_t c0_b = __shfl_sync(mask, cw_b, base);
        std::uint8_t* slot2_a =
            scratch_a + static_cast<std::size_t>(c0_a & byte_address_mask);
        std::uint8_t* slot2_b =
            scratch_b + static_cast<std::size_t>(c0_b & byte_address_mask);

        const auto* t_addr_a =
            reinterpret_cast<const std::uint32_t*>(slot2_a) + lane;
        const auto* t_addr_b =
            reinterpret_cast<const std::uint32_t*>(slot2_b) + lane;
        std::uint32_t t_word_a = 0U;
        std::uint32_t t_word_b = 0U;
        asm volatile(
            "ld.global.u32 %0, [%2];\n\t"
            "ld.global.u32 %1, [%3];\n\t"
            : "=r"(t_word_a), "=r"(t_word_b)
            : "l"(t_addr_a), "l"(t_addr_b));

        const std::uint32_t c1_a =
            __shfl_sync(mask, cw_a, base + 1U);
        const std::uint32_t c1_b =
            __shfl_sync(mask, cw_b, base + 1U);
        const std::uint32_t t0_hi_a =
            __shfl_sync(mask, t_word_a, base + 1U);
        const std::uint32_t t0_hi_b =
            __shfl_sync(mask, t_word_b, base + 1U);
        const std::uint32_t old_a2_a =
            __shfl_sync(mask, aw_a, base + 2U);
        const std::uint32_t old_a3_a =
            __shfl_sync(mask, aw_a, base + 3U);
        const std::uint32_t old_a2_b =
            __shfl_sync(mask, aw_b, base + 2U);
        const std::uint32_t old_a3_b =
            __shfl_sync(mask, aw_b, base + 3U);

        std::uint32_t na0_lo_a = 0U, na0_hi_a = 0U;
        std::uint32_t na1_lo_a = 0U, na1_hi_a = 0U;
        std::uint32_t na0_lo_b = 0U, na0_hi_b = 0U;
        std::uint32_t na1_lo_b = 0U, na1_hi_b = 0U;

        if (lane == 0U) {
            const std::uint32_t t0_lo_a = t_word_a;
            const std::uint64_t c_low_a =
                static_cast<std::uint64_t>(c0_a) |
                (static_cast<std::uint64_t>(c1_a) << 32U);
            const std::uint64_t t0v_a =
                static_cast<std::uint64_t>(t0_lo_a) |
                (static_cast<std::uint64_t>(t0_hi_a) << 32U);
            const std::uint64_t lo_a = c_low_a * t0v_a;
            const std::uint64_t hi_a = __umul64hi(c_low_a, t0v_a);
            const std::uint64_t olda0_a =
                static_cast<std::uint64_t>(aw0_a) |
                (static_cast<std::uint64_t>(aw1_a) << 32U);
            const std::uint64_t olda1_a =
                static_cast<std::uint64_t>(old_a2_a) |
                (static_cast<std::uint64_t>(old_a3_a) << 32U);
            const std::uint64_t next0_a = olda0_a + hi_a;
            const std::uint64_t next1_a = olda1_a + lo_a;
            na0_lo_a = static_cast<std::uint32_t>(next0_a);
            na0_hi_a = static_cast<std::uint32_t>(next0_a >> 32U);
            na1_lo_a = static_cast<std::uint32_t>(next1_a);
            na1_hi_a = static_cast<std::uint32_t>(next1_a >> 32U);
            aw1_shadow_a = na0_hi_a ^ t0_hi_a;

            const std::uint32_t t0_lo_b = t_word_b;
            const std::uint64_t c_low_b =
                static_cast<std::uint64_t>(c0_b) |
                (static_cast<std::uint64_t>(c1_b) << 32U);
            const std::uint64_t t0v_b =
                static_cast<std::uint64_t>(t0_lo_b) |
                (static_cast<std::uint64_t>(t0_hi_b) << 32U);
            const std::uint64_t lo_b = c_low_b * t0v_b;
            const std::uint64_t hi_b = __umul64hi(c_low_b, t0v_b);
            const std::uint64_t olda0_b =
                static_cast<std::uint64_t>(aw0_b) |
                (static_cast<std::uint64_t>(aw1_b) << 32U);
            const std::uint64_t olda1_b =
                static_cast<std::uint64_t>(old_a2_b) |
                (static_cast<std::uint64_t>(old_a3_b) << 32U);
            const std::uint64_t next0_b = olda0_b + hi_b;
            const std::uint64_t next1_b = olda1_b + lo_b;
            na0_lo_b = static_cast<std::uint32_t>(next0_b);
            na0_hi_b = static_cast<std::uint32_t>(next0_b >> 32U);
            na1_lo_b = static_cast<std::uint32_t>(next1_b);
            na1_hi_b = static_cast<std::uint32_t>(next1_b >> 32U);
            aw1_shadow_b = na0_hi_b ^ t0_hi_b;
        }

        na0_lo_a = __shfl_sync(mask, na0_lo_a, base);
        na0_hi_a = __shfl_sync(mask, na0_hi_a, base);
        na1_lo_a = __shfl_sync(mask, na1_lo_a, base);
        na1_hi_a = __shfl_sync(mask, na1_hi_a, base);
        na0_lo_b = __shfl_sync(mask, na0_lo_b, base);
        na0_hi_b = __shfl_sync(mask, na0_hi_b, base);
        na1_lo_b = __shfl_sync(mask, na1_lo_b, base);
        na1_hi_b = __shfl_sync(mask, na1_hi_b, base);

        const std::uint32_t na_word_a =
            lane == 0U ? na0_lo_a :
            lane == 1U ? na0_hi_a :
            lane == 2U ? na1_lo_a : na1_hi_a;
        const std::uint32_t na_word_b =
            lane == 0U ? na0_lo_b :
            lane == 1U ? na0_hi_b :
            lane == 2U ? na1_lo_b : na1_hi_b;

        std::uint32_t store_a = na_word_a;
        if (lane == 2U) store_a ^= tweak_lo_a;
        if (lane == 3U) store_a ^= tweak_hi_a;
        reinterpret_cast<std::uint32_t*>(slot2_a)[lane] = store_a;

        if (have_b) {
            std::uint32_t store_b = na_word_b;
            if (lane == 2U) store_b ^= tweak_lo_b;
            if (lane == 3U) store_b ^= tweak_hi_b;
            reinterpret_cast<std::uint32_t*>(slot2_b)[lane] = store_b;
        }

        aw_a = na_word_a ^ t_word_a;
        bw_a = cw_a;
        if (have_b) {
            aw_b = na_word_b ^ t_word_b;
            bw_b = cw_b;
        }
    }
}


template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_readonly(std::size_t count,
                                                         std::uint8_t* __restrict__ scratchpads,
                                                         const cryptonight::SplitContext* __restrict__ contexts)
{
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

    const std::uint32_t* t0 = g_cn_aes_ttables_readonly;
    const std::uint32_t* t1 = g_cn_aes_ttables_readonly + 256;
    const std::uint32_t* t2 = g_cn_aes_ttables_readonly + 512;
    const std::uint32_t* t3 = g_cn_aes_ttables_readonly + 768;

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
            const std::uint32_t q0 = cn_aes_readonly_load(t0, static_cast<std::uint8_t>(own));
            const std::uint32_t q1_local = cn_aes_readonly_load(t1, static_cast<std::uint8_t>(own >> 8U));
            const std::uint32_t q2_local = cn_aes_readonly_load(t2, static_cast<std::uint8_t>(own >> 16U));
            const std::uint32_t q3_local = cn_aes_readonly_load(t3, static_cast<std::uint8_t>(own >> 24U));
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
            cw = cn_aes_readonly_load(t0, x0) ^ cn_aes_readonly_load(t1, x1) ^ cn_aes_readonly_load(t2, x2) ^ cn_aes_readonly_load(t3, x3) ^ aw;
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


template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable2_coalesced(
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

    std::uint8_t* scratch =
        scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];

    std::uint64_t a =
        reinterpret_cast<const std::uint64_t*>(ctx.a)[lane];
    std::uint64_t b =
        reinterpret_cast<const std::uint64_t*>(ctx.b)[lane];
    const std::uint64_t tweak = ctx.tweak;

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint64_t a0 = __shfl_sync(mask, a, base);
        const std::uint32_t byte_offset =
            static_cast<std::uint32_t>(a0) & byte_address_mask;
        auto* slot_words =
            reinterpret_cast<std::uint64_t*>(
                scratch + static_cast<std::size_t>(byte_offset));

        const std::uint64_t own = slot_words[lane];
        const std::uint64_t other =
            __shfl_sync(mask, own, base + (lane ^ 1U));

        const std::uint32_t own0 = static_cast<std::uint32_t>(own);
        const std::uint32_t own1 = static_cast<std::uint32_t>(own >> 32U);
        const std::uint32_t other0 = static_cast<std::uint32_t>(other);
        const std::uint32_t other1 = static_cast<std::uint32_t>(other >> 32U);

        std::uint32_t cw0 = 0U;
        std::uint32_t cw1 = 0U;
        const std::uint32_t a_lo = static_cast<std::uint32_t>(a);
        const std::uint32_t a_hi = static_cast<std::uint32_t>(a >> 32U);

        if (lane == 0U) {
            // own = w0|w1, other = w2|w3. Produce AES words c0/c1.
            cw0 =
                t0[static_cast<std::uint8_t>(own0)] ^
                t1[static_cast<std::uint8_t>(own1 >> 8U)] ^
                t2[static_cast<std::uint8_t>(other0 >> 16U)] ^
                t3[static_cast<std::uint8_t>(other1 >> 24U)] ^
                a_lo;
            cw1 =
                t0[static_cast<std::uint8_t>(own1)] ^
                t1[static_cast<std::uint8_t>(other0 >> 8U)] ^
                t2[static_cast<std::uint8_t>(other1 >> 16U)] ^
                t3[static_cast<std::uint8_t>(own0 >> 24U)] ^
                a_hi;
        } else {
            // own = w2|w3, other = w0|w1. Produce AES words c2/c3.
            cw0 =
                t0[static_cast<std::uint8_t>(own0)] ^
                t1[static_cast<std::uint8_t>(own1 >> 8U)] ^
                t2[static_cast<std::uint8_t>(other0 >> 16U)] ^
                t3[static_cast<std::uint8_t>(other1 >> 24U)] ^
                a_lo;
            cw1 =
                t0[static_cast<std::uint8_t>(own1)] ^
                t1[static_cast<std::uint8_t>(other0 >> 8U)] ^
                t2[static_cast<std::uint8_t>(other1 >> 16U)] ^
                t3[static_cast<std::uint8_t>(own0 >> 24U)] ^
                a_hi;
        }

        const std::uint64_t c =
            static_cast<std::uint64_t>(cw0) |
            (static_cast<std::uint64_t>(cw1) << 32U);

        std::uint64_t first_store = c ^ b;
        if (lane == 1U) {
            // Variant-1 mutation is byte 11: high byte of word 2, which is
            // the high byte of this lane's low 32-bit word.
            std::uint32_t low = static_cast<std::uint32_t>(first_store);
            const std::uint8_t tmp = static_cast<std::uint8_t>(low >> 24U);
            constexpr std::uint32_t table = 0x75310U;
            const std::uint8_t mutate_index =
                static_cast<std::uint8_t>(
                    ((((tmp >> 3U) & 6U) | (tmp & 1U)) << 1U));
            low ^= ((table >> mutate_index) & 0x30U) << 24U;
            first_store =
                (first_store & 0xffffffff00000000ULL) |
                static_cast<std::uint64_t>(low);
        }
        slot_words[lane] = first_store;

        const std::uint64_t c_low = __shfl_sync(mask, c, base);
        const std::uint32_t byte_offset2 =
            static_cast<std::uint32_t>(c_low) & byte_address_mask;
        auto* slot2_words =
            reinterpret_cast<std::uint64_t*>(
                scratch + static_cast<std::size_t>(byte_offset2));

        const std::uint64_t t = slot2_words[lane];
        const std::uint64_t t0v = __shfl_sync(mask, t, base);
        const std::uint64_t old_a1 = __shfl_sync(mask, a, base + 1U);

        std::uint64_t next0 = 0U;
        std::uint64_t next1 = 0U;
        if (lane == 0U) {
            const std::uint64_t lo = c_low * t0v;
            const std::uint64_t hi = __umul64hi(c_low, t0v);
            next0 = a0 + hi;
            next1 = old_a1 + lo;
        }
        next0 = __shfl_sync(mask, next0, base);
        next1 = __shfl_sync(mask, next1, base);

        slot2_words[lane] =
            lane == 0U ? next0 : (next1 ^ tweak);
        a = (lane == 0U ? next0 : next1) ^ t;
        b = c;
    }
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_mul4(std::size_t count,
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

    // Distributed-multiply candidate: every lane keeps its own A word live.
    // The 64x64->128 multiply/add is decomposed into four 32x32 partial
    // products, one per lane, so lane 0 no longer owns the whole critical path.

    const std::uint64_t tweak = ctx.tweak;
    const std::uint32_t tweak_lo = static_cast<std::uint32_t>(tweak);
    const std::uint32_t tweak_hi = static_cast<std::uint32_t>(tweak >> 32U);

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        // ((a_low >> 4) & address_mask) * 16 is exactly the low word masked
        // with (address_mask << 4). This removes a subgroup shuffle, 64-bit
        // combine and widened multiply from the first dependent address path.
        const std::uint32_t aw0 = __shfl_sync(mask, aw, base);
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

        // Map the four 32x32 partial products onto the four subgroup lanes:
        //   lane 2: p00 = c0 * t0_lo
        //   lane 3: p01 = c0 * t0_hi
        //   lane 0: p10 = c1 * t0_lo
        //   lane 1: p11 = c1 * t0_hi
        // The target mapping is chosen so each lane can finish the A word it
        // already owns after two compact carry exchanges.
        const unsigned int t_source = base + (lane & 1U);
        const std::uint32_t t_half =
            __shfl_sync(mask, t_word, t_source);
        const unsigned int c_source =
            base + (lane < 2U ? 1U : 0U);
        const std::uint32_t c_half =
            __shfl_sync(mask, cw, c_source);

        const std::uint32_t product_lo = c_half * t_half;
        const std::uint32_t product_hi = __umulhi(c_half, t_half);

        // In one low-word shuffle, lane 0 receives p11.lo and lane 3 receives
        // p10.lo. In one high-word shuffle, lane 0 receives p01.hi and lane 3
        // receives p00.hi. Other lanes simply read themselves.
        const unsigned int lo_source =
            lane == 0U ? base + 1U :
            lane == 3U ? base + 0U :
                         base + lane;
        const unsigned int hi_source =
            lane == 0U ? base + 3U :
            lane == 3U ? base + 2U :
                         base + lane;
        const std::uint32_t peer_lo =
            __shfl_sync(mask, product_lo, lo_source);
        const std::uint32_t peer_hi =
            __shfl_sync(mask, product_hi, hi_source);

        std::uint32_t word1 = 0U;
        std::uint32_t carry1 = 0U;
        std::uint32_t carry_b = 0U;
        std::uint32_t na_word = 0U;

        if (lane == 3U) {
            std::uint32_t s = peer_hi + product_lo;
            carry1 = s < peer_hi ? 1U : 0U;
            const std::uint32_t s2 = s + peer_lo;
            carry1 += s2 < s ? 1U : 0U;
            word1 = s2;
        } else if (lane == 2U) {
            const std::uint32_t sum = aw + product_lo;
            carry_b = sum < aw ? 1U : 0U;
            na_word = sum;
        }

        // One shuffle simultaneously gives lane 0 the product carry from lane 3
        // and lane 3 the low-64 add carry from lane 2.
        const std::uint32_t carry_seed =
            lane == 3U ? carry1 :
            lane == 2U ? carry_b : 0U;
        const unsigned int carry_source =
            lane == 0U ? base + 3U :
            lane == 3U ? base + 2U :
                         base + lane;
        const std::uint32_t carried =
            __shfl_sync(mask, carry_seed, carry_source);

        std::uint32_t packed_hi_carries = 0U;
        if (lane == 0U) {
            std::uint32_t s = product_hi + peer_hi;
            std::uint32_t carry2 = s < product_hi ? 1U : 0U;
            const std::uint32_t s2 = s + peer_lo;
            carry2 += s2 < s ? 1U : 0U;
            const std::uint32_t s3 = s2 + carried;
            carry2 += s3 < s2 ? 1U : 0U;

            const std::uint32_t sum = aw + s3;
            const std::uint32_t carry_a = sum < aw ? 1U : 0U;
            na_word = sum;
            packed_hi_carries = carry2 | (carry_a << 2U);
        } else if (lane == 3U) {
            na_word = aw + word1 + carried;
        }

        // Lane 1 finishes the high word of the high 64-bit half. Pack carry2
        // and carry_a together so only one subgroup shuffle is required.
        const std::uint32_t hi_carries =
            __shfl_sync(mask, packed_hi_carries, base);
        if (lane == 1U) {
            const std::uint32_t word3 =
                product_hi + (hi_carries & 3U);
            na_word = aw + word3 + ((hi_carries >> 2U) & 1U);
        }

        std::uint32_t store_word = na_word;
        if (lane == 2U) store_word ^= tweak_lo;
        if (lane == 3U) store_word ^= tweak_hi;
        reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

        aw = na_word ^ t_word;
        bw = cw;
    }
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable4_pairload(std::size_t count,
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

        // Split the warp's eight independent 4-lane hashes into two cohorts.
        // The inline PTX deliberately issues the two half-warp random loads into
        // different destination registers before either cohort consumes its
        // result. This avoids one warp-wide scoreboard entry making four ready
        // hashes wait for a slower miss in the other four.
        const unsigned int cohort = (warp_lane >> 2U) & 1U;
        const auto* own_address =
            reinterpret_cast<const std::uint32_t*>(slot) + lane;
        std::uint32_t own_even = 0U;
        std::uint32_t own_odd = 0U;
        asm volatile(
            "{ .reg .pred p;\n\t"
            "setp.eq.u32 p, %2, 0;\n\t"
            "@p ld.global.u32 %0, [%3];\n\t"
            "@!p ld.global.u32 %1, [%3];\n\t"
            "}\n"
            : "+r"(own_even), "+r"(own_odd)
            : "r"(cohort), "l"(own_address));

        std::uint32_t cw = 0U;
        if constexpr (VariantIndex == 2) {
            if (cohort == 0U) {
                const std::uint32_t q0 = t0[static_cast<std::uint8_t>(own_even)];
                const std::uint32_t q1_local = t1[static_cast<std::uint8_t>(own_even >> 8U)];
                const std::uint32_t q2_local = t2[static_cast<std::uint8_t>(own_even >> 16U)];
                const std::uint32_t q3_local = t3[static_cast<std::uint8_t>(own_even >> 24U)];
                const std::uint32_t q1 = __shfl_sync(mask, q1_local, base + ((lane + 1U) & 3U));
                const std::uint32_t q2 = __shfl_sync(mask, q2_local, base + ((lane + 2U) & 3U));
                const std::uint32_t q3 = __shfl_sync(mask, q3_local, base + ((lane + 3U) & 3U));
                cw = q0 ^ q1 ^ q2 ^ q3 ^ aw;
            } else {
                const std::uint32_t q0 = t0[static_cast<std::uint8_t>(own_odd)];
                const std::uint32_t q1_local = t1[static_cast<std::uint8_t>(own_odd >> 8U)];
                const std::uint32_t q2_local = t2[static_cast<std::uint8_t>(own_odd >> 16U)];
                const std::uint32_t q3_local = t3[static_cast<std::uint8_t>(own_odd >> 24U)];
                const std::uint32_t q1 = __shfl_sync(mask, q1_local, base + ((lane + 1U) & 3U));
                const std::uint32_t q2 = __shfl_sync(mask, q2_local, base + ((lane + 2U) & 3U));
                const std::uint32_t q3 = __shfl_sync(mask, q3_local, base + ((lane + 3U) & 3U));
                cw = q0 ^ q1 ^ q2 ^ q3 ^ aw;
            }
        } else {
            if (cohort == 0U) {
                const std::uint32_t w1 = __shfl_sync(mask, own_even, base + ((lane + 1U) & 3U));
                const std::uint32_t w2 = __shfl_sync(mask, own_even, base + ((lane + 2U) & 3U));
                const std::uint32_t w3 = __shfl_sync(mask, own_even, base + ((lane + 3U) & 3U));
                const std::uint8_t x0 = static_cast<std::uint8_t>(own_even);
                const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
                const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
                const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);
                cw = t0[x0] ^ t1[x1] ^ t2[x2] ^ t3[x3] ^ aw;
            } else {
                const std::uint32_t w1 = __shfl_sync(mask, own_odd, base + ((lane + 1U) & 3U));
                const std::uint32_t w2 = __shfl_sync(mask, own_odd, base + ((lane + 2U) & 3U));
                const std::uint32_t w3 = __shfl_sync(mask, own_odd, base + ((lane + 3U) & 3U));
                const std::uint8_t x0 = static_cast<std::uint8_t>(own_odd);
                const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
                const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
                const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);
                cw = t0[x0] ^ t1[x1] ^ t2[x2] ^ t3[x3] ^ aw;
            }
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


} // anonymous namespace