// 64-byte aligned scratchpad-tile CryptoNight phase-2 candidate.
//
// This mirrors the mature CUDA CryptoNight memory model: two lanes cooperate on
// one hash, move a 64-byte aligned scratchpad region into per-hash shared memory,
// operate on the addressed 16-byte quarter, then write the region back. Yerbas'
// exact variant-1 mutation and multiply/add dependency chain are unchanged.
//
// The candidate is never authoritative by itself. The production selector
// validates its complete scratchpad result against the promoted 4-lane T-table
// baseline before timing it.

namespace {

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable2_tile64(
    std::size_t count,
    std::uint8_t* scratchpads,
    const cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t tables[1024];
    extern __shared__ std::uint64_t tile_mem[];
    cryptonight::init_shared_aes_ttables(tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;

    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 1U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~1U;
    const unsigned int group = global_thread >> 1U;
    const unsigned int local_group = threadIdx.x >> 1U;
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
    std::uint64_t* tile = tile_mem + static_cast<std::size_t>(local_group) * 8U;

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const std::uint64_t a0 = __shfl_sync(mask, a, base);
        const std::uint32_t index = static_cast<std::uint32_t>((a0 >> 4U) & address_mask);
        const std::uint32_t quarter = index & 3U;
        std::uint8_t* tile_base = scratch + static_cast<std::size_t>(index & ~3U) * 16U;

        // Each lane performs four naturally contiguous 64-bit loads. Together
        // the pair moves one aligned 64-byte region into shared memory.
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            tile[lane * 4U + static_cast<unsigned int>(k)] =
                reinterpret_cast<const std::uint64_t*>(tile_base)[lane * 4U + static_cast<unsigned int>(k)];
        __syncwarp(mask);

        std::uint64_t* selected = tile + static_cast<std::size_t>(quarter) * 2U;
        const std::uint64_t s = selected[lane];
        const std::uint64_t other = __shfl_sync(mask, s, base + (lane ^ 1U));
        const std::uint8_t* sb = reinterpret_cast<const std::uint8_t*>(&s);
        const std::uint8_t* ob = reinterpret_cast<const std::uint8_t*>(&other);

        const std::uint32_t cw_lo =
            t0[sb[0]] ^ t1[sb[5]] ^ t2[ob[2]] ^ t3[ob[7]] ^ static_cast<std::uint32_t>(a);
        const std::uint32_t cw_hi =
            t0[sb[4]] ^ t1[ob[1]] ^ t2[ob[6]] ^ t3[sb[3]] ^ static_cast<std::uint32_t>(a >> 32U);
        const std::uint64_t c = static_cast<std::uint64_t>(cw_lo) |
                                (static_cast<std::uint64_t>(cw_hi) << 32U);

        selected[lane] = c ^ b;
        __syncwarp(mask);
        if (lane == 0U)
            cryptonight::variant1_mutate(reinterpret_cast<std::uint8_t*>(selected));
        __syncwarp(mask);

        #pragma unroll
        for (int k = 0; k < 4; ++k)
            reinterpret_cast<std::uint64_t*>(tile_base)[lane * 4U + static_cast<unsigned int>(k)] =
                tile[lane * 4U + static_cast<unsigned int>(k)];
        __syncwarp(mask);

        const std::uint64_t c0 = __shfl_sync(mask, c, base);
        const std::uint32_t j = static_cast<std::uint32_t>((c0 >> 4U) & address_mask);
        const std::uint32_t quarter2 = j & 3U;
        std::uint8_t* tile_base2 = scratch + static_cast<std::size_t>(j & ~3U) * 16U;

        #pragma unroll
        for (int k = 0; k < 4; ++k)
            tile[lane * 4U + static_cast<unsigned int>(k)] =
                reinterpret_cast<const std::uint64_t*>(tile_base2)[lane * 4U + static_cast<unsigned int>(k)];
        __syncwarp(mask);

        std::uint64_t* selected2 = tile + static_cast<std::size_t>(quarter2) * 2U;
        const std::uint64_t tv = selected2[lane];
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

        selected2[lane] = lane == 0U ? next0 : (next1 ^ tweak);
        __syncwarp(mask);
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            reinterpret_cast<std::uint64_t*>(tile_base2)[lane * 4U + static_cast<unsigned int>(k)] =
                tile[lane * 4U + static_cast<unsigned int>(k)];

        a = (lane == 0U ? next0 : next1) ^ tv;
        b = c;
        __syncwarp(mask);
    }
}

inline std::size_t cn_tile64_dynamic_shared_bytes(int threads)
{
    // One 64-byte tile for every pair of CUDA threads.
    return static_cast<std::size_t>(threads / 2) * 64U;
}

} // anonymous namespace
