// Compact-broadcast CryptoNight phase-2 candidate.
//
// Keep the historical mode-444 kernel symbol so the existing production selector
// can parity-test and benchmark this candidate without changing routing or cache
// formats. The previous cooperative-multiply experiment split the 64x64 product
// across lanes 0/1 but lost on measured production batches. This candidate keeps
// the proven baseline arithmetic entirely in lane 0 and only removes subgroup
// traffic that the owning lane does not need: t0_lo stays local to lane 0 and
// na0_lo is consumed locally instead of being broadcast back to its producer.
//
// The production selector still requires exact full-scratchpad parity and a
// measured production-batch win before this kernel can be promoted.

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

    const std::uint32_t* table0 = tables;
    const std::uint32_t* table1 = tables + 256;
    const std::uint32_t* table2 = tables + 512;
    const std::uint32_t* table3 = tables + 768;

    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];
    std::uint32_t aw = cryptonight::load32le_aligned(ctx.a + lane * 4U);
    std::uint32_t bw = cryptonight::load32le_aligned(ctx.b + lane * 4U);

    // CN-Fast's dependent address only consumes the low 32-bit A word. Lane 0
    // carries the high word needed later by the multiply/add step so the next
    // iteration does not have to place that shuffle on the address path.
    std::uint32_t fast_aw1_shadow = 0U;
    if constexpr (VariantIndex == 2) {
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
            aw0 = __shfl_sync(mask, aw, base);
            const std::uint32_t byte_offset = aw0 & byte_address_mask;
            slot = scratch + static_cast<std::size_t>(byte_offset);
            if (lane == 0U) aw1 = fast_aw1_shadow;
        } else {
            aw0 = __shfl_sync(mask, aw, base);
            aw1 = __shfl_sync(mask, aw, base + 1U);
            const std::uint64_t a_low = static_cast<std::uint64_t>(aw0) |
                                        (static_cast<std::uint64_t>(aw1) << 32U);
            const std::uint32_t index =
                static_cast<std::uint32_t>((a_low >> 4U) & address_mask);
            slot = scratch + static_cast<std::size_t>(index) * 16U;
        }

        auto* slot_words = reinterpret_cast<std::uint32_t*>(slot);
        const std::uint32_t own = slot_words[lane];

        std::uint32_t cw = 0U;
        if constexpr (VariantIndex == 2) {
            // Issue all four independent shared-table reads from each lane's
            // local word before exchanging their ShiftRows contributions. This
            // gives the scheduler more independent work while the dependency-
            // heavy scratchpad chain is waiting.
            const std::uint32_t q0 = table0[static_cast<std::uint8_t>(own)];
            const std::uint32_t q1_local = table1[static_cast<std::uint8_t>(own >> 8U)];
            const std::uint32_t q2_local = table2[static_cast<std::uint8_t>(own >> 16U)];
            const std::uint32_t q3_local = table3[static_cast<std::uint8_t>(own >> 24U)];
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
            const std::uint8_t x0 = static_cast<std::uint8_t>(own);
            const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
            const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
            const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);
            cw = table0[x0] ^ table1[x1] ^ table2[x2] ^ table3[x3] ^ aw;
        }

        // Variant-1 mutation touches byte 11, which lives in lane 2's high byte.
        std::uint32_t first_store_word = cw ^ bw;
        if (lane == 2U) {
            const std::uint8_t tmp = static_cast<std::uint8_t>(first_store_word >> 24U);
            constexpr std::uint32_t table = 0x75310U;
            const std::uint8_t mutate_index =
                static_cast<std::uint8_t>((((tmp >> 3U) & 6U) | (tmp & 1U)) << 1U);
            first_store_word ^= ((table >> mutate_index) & 0x30U) << 24U;
        }
        slot_words[lane] = first_store_word;

        std::uint32_t c0 = 0U;
        std::uint32_t c1 = 0U;
        std::uint8_t* slot2 = nullptr;
        if constexpr (VariantIndex == 2) {
            c0 = __shfl_sync(mask, cw, base);
            const std::uint32_t byte_offset = c0 & byte_address_mask;
            slot2 = scratch + static_cast<std::size_t>(byte_offset);
        } else {
            c0 = __shfl_sync(mask, cw, base);
            c1 = __shfl_sync(mask, cw, base + 1U);
            const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                        (static_cast<std::uint64_t>(c1) << 32U);
            const std::uint32_t j =
                static_cast<std::uint32_t>((c_low >> 4U) & address_mask);
            slot2 = scratch + static_cast<std::size_t>(j) * 16U;
        }

        auto* slot2_words = reinterpret_cast<std::uint32_t*>(slot2);
        const std::uint32_t t_word = slot2_words[lane];

        // Restore the baseline's lane-0 ownership of the multiply inputs. The
        // previous split-multiply candidate had to broadcast t0_lo to lane 1;
        // this serial candidate does not, so lane 0 consumes its own load.
        const std::uint32_t t0_lo = lane == 0U ? t_word : 0U;
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        if constexpr (VariantIndex == 2) {
            c1 = __shfl_sync(mask, cw, base + 1U);
        }
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t na0_lo_local = 0U;
        std::uint32_t na0_hi_local = 0U;
        std::uint32_t na1_lo_local = 0U;
        std::uint32_t na1_hi_local = 0U;
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
            na0_lo_local = static_cast<std::uint32_t>(na0);
            na0_hi_local = static_cast<std::uint32_t>(na0 >> 32U);
            na1_lo_local = static_cast<std::uint32_t>(na1);
            na1_hi_local = static_cast<std::uint32_t>(na1 >> 32U);
            if constexpr (VariantIndex == 2) {
                fast_aw1_shadow = na0_hi_local ^ t0_hi;
            }
        }

        // Lane 0 already owns na0_lo and consumes it locally. Only the three
        // words needed by lanes 1..3 cross the subgroup, removing one result
        // broadcast from every dependency-heavy iteration.
        const std::uint32_t na0_hi = __shfl_sync(mask, na0_hi_local, base);
        const std::uint32_t na1_lo = __shfl_sync(mask, na1_lo_local, base);
        const std::uint32_t na1_hi = __shfl_sync(mask, na1_hi_local, base);
        const std::uint32_t na_word = lane == 0U ? na0_lo_local :
                                      lane == 1U ? na0_hi :
                                      lane == 2U ? na1_lo : na1_hi;
        const std::uint32_t store_word = lane == 2U ? (na_word ^ tweak_lo) :
                                         lane == 3U ? (na_word ^ tweak_hi) : na_word;
        slot2_words[lane] = store_word;

        aw = na_word ^ t_word;
        bw = cw;
    }
}

} // anonymous namespace
