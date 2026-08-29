// Cache-global CryptoNight phase-2 candidate.
//
// This is deliberately architecture-generic: every supported CUDA target builds
// both the ordinary global-memory T-table kernel and this cache-global variant.
// The production selector decides per GPU/variant from exact parity plus measured
// timing; there are no GPU model or compute-capability promotion rules.
//
// PTX .cg accesses cache global scratchpad traffic at L2 rather than allocating
// the random CryptoNight working set in L1. Context/state traffic remains normal.

namespace {

__device__ __forceinline__ std::uint32_t cn_global_cg_load32(const std::uint32_t* ptr)
{
    std::uint32_t value;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
#else
    value = *ptr;
#endif
    return value;
}

__device__ __forceinline__ void cn_global_cg_store32(std::uint32_t* ptr, std::uint32_t value)
{
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
#else
    *ptr = value;
#endif
}

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

        // Four lanes issue one contiguous 32-bit load each. The cache-global
        // modifier prevents the random scratchpad stream from displacing useful
        // L1 data while retaining the proven coalesced + shuffle AES layout.
        const std::uint32_t own = cn_global_cg_load32(slot_words + lane);
        const std::uint32_t w1 = __shfl_sync(mask, own, base + ((lane + 1U) & 3U));
        const std::uint32_t w2 = __shfl_sync(mask, own, base + ((lane + 2U) & 3U));
        const std::uint32_t w3 = __shfl_sync(mask, own, base + ((lane + 3U) & 3U));
        const std::uint8_t x0 = static_cast<std::uint8_t>(own);
        const std::uint8_t x1 = static_cast<std::uint8_t>(w1 >> 8U);
        const std::uint8_t x2 = static_cast<std::uint8_t>(w2 >> 16U);
        const std::uint8_t x3 = static_cast<std::uint8_t>(w3 >> 24U);
        const std::uint32_t cw = table0[x0] ^ table1[x1] ^ table2[x2] ^ table3[x3] ^ aw;

        cn_global_cg_store32(slot_words + lane, cw ^ bw);
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

        const std::uint32_t t_word = cn_global_cg_load32(slot2_words + lane);
        const std::uint32_t t0_lo = __shfl_sync(mask, t_word, base);
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        const std::uint64_t t0v = static_cast<std::uint64_t>(t0_lo) |
                                  (static_cast<std::uint64_t>(t0_hi) << 32U);
        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t na0_lo = 0U, na0_hi = 0U, na1_lo = 0U, na1_hi = 0U;
        if (lane == 0U) {
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
        }
        na0_lo = __shfl_sync(mask, na0_lo, base);
        na0_hi = __shfl_sync(mask, na0_hi, base);
        na1_lo = __shfl_sync(mask, na1_lo, base);
        na1_hi = __shfl_sync(mask, na1_hi, base);

        const std::uint32_t store_word = lane == 0U ? na0_lo :
                                         lane == 1U ? na0_hi :
                                         lane == 2U ? (na1_lo ^ tweak_lo) : (na1_hi ^ tweak_hi);
        cn_global_cg_store32(slot2_words + lane, store_word);

        const std::uint32_t na_word = lane == 0U ? na0_lo :
                                      lane == 1U ? na0_hi :
                                      lane == 2U ? na1_lo : na1_hi;
        aw = na_word ^ t_word;
        bw = cw;
        __syncwarp(mask);
    }
}

} // anonymous namespace
