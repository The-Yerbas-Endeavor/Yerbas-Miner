// Production cooperative CryptoNight pipeline.
//
// CryptoNight v1 contains three expensive phases.  The dependency-heavy memory
// loop uses four adjacent CUDA lanes per hash.  Setup and finalization contain
// eight independent 16-byte AES chains, so those phases use eight adjacent CUDA
// lanes per hash: one lane per AES block.  This preserves the exact CryptoNight
// byte order while exposing the natural block-level parallelism that the scalar
// one-thread-per-hash setup/final kernels leave unused.
//
// The design is generic CUDA: no GPU model or compute-capability selection is
// used.  Whole-GhostRider startup parity remains the production correctness
// gate for the complete pipeline.

namespace {

__device__ __forceinline__ std::uint32_t cn_coop_mix_column(std::uint8_t a0,
                                                            std::uint8_t a1,
                                                            std::uint8_t a2,
                                                            std::uint8_t a3)
{
    const std::uint8_t t = static_cast<std::uint8_t>(a0 ^ a1 ^ a2 ^ a3);
    const std::uint8_t o0 = static_cast<std::uint8_t>(a0 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a0 ^ a1)));
    const std::uint8_t o1 = static_cast<std::uint8_t>(a1 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a1 ^ a2)));
    const std::uint8_t o2 = static_cast<std::uint8_t>(a2 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a2 ^ a3)));
    const std::uint8_t o3 = static_cast<std::uint8_t>(a3 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a3 ^ a0)));
    return cryptonight::pack4(o0, o1, o2, o3);
}

template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_setup_stage_cooperative8(std::uint8_t* states,
                                                      std::size_t count,
                                                      std::uint8_t* scratchpads,
                                                      cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int lane = warp_lane & 7U;
    const unsigned int base = warp_lane & ~7U;
    const unsigned int group = global_thread >> 3U;
    if (group >= count) return;
    const unsigned int mask = 0xFFU << base;

    std::uint8_t* state = states + static_cast<std::size_t>(group) * kStateBytes;
    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    auto& ctx = contexts[group];

    // Keccak and A/B/tweak derivation are true per-hash work.  One lane performs
    // them once, then the subgroup fans out over the eight independent AES
    // chains that fill the scratchpad.
    if (lane == 0U) {
        cryptonight::keccak1600(state, kStateBytes, ctx.state);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            ctx.a[i] = static_cast<std::uint8_t>(ctx.state[i] ^ ctx.state[32 + i]);
            ctx.b[i] = static_cast<std::uint8_t>(ctx.state[16 + i] ^ ctx.state[48 + i]);
        }
        ctx.tweak = cryptonight::cn_load64(state + 35) ^
                    cryptonight::cn_load64_aligned(ctx.state + 192);
    }
    __syncwarp(mask);

    alignas(16) std::uint8_t text[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        text[i] = ctx.state[64 + static_cast<int>(lane) * 16 + i];

    // Key expansion is inexpensive relative to thousands of AES pseudo-rounds;
    // duplicating it per cooperating lane avoids shared-memory key traffic and
    // keeps each independent AES chain self-contained.
    alignas(16) std::uint8_t expanded[240];
    cryptonight::aes256_expand_key(ctx.state, expanded);

    #pragma unroll 1
    for (std::size_t round = 0; round < init_rounds; ++round) {
        if constexpr (UseTTable)
            cryptonight::aes_pseudo_round_ttable(text, expanded, aes_tables);
        else
            cryptonight::aes_pseudo_round(text, expanded);

        std::uint8_t* out = scratch + round * 128U + static_cast<std::size_t>(lane) * 16U;
        #pragma unroll
        for (int i = 0; i < 16; ++i) out[i] = text[i];
    }
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_cooperative(std::size_t count,
                                                   std::uint8_t* scratchpads,
                                                   const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~3U;
    const unsigned int group = global_thread >> 2;
    if (group >= count) return;
    const unsigned int mask = 0xFU << base;

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

        std::uint8_t x0, x1, x2, x3;
        if (lane == 0U) { x0 = slot[0];  x1 = slot[5];  x2 = slot[10]; x3 = slot[15]; }
        else if (lane == 1U) { x0 = slot[4];  x1 = slot[9];  x2 = slot[14]; x3 = slot[3]; }
        else if (lane == 2U) { x0 = slot[8];  x1 = slot[13]; x2 = slot[2];  x3 = slot[7]; }
        else { x0 = slot[12]; x1 = slot[1]; x2 = slot[6]; x3 = slot[11]; }

        const std::uint32_t cw = cn_coop_mix_column(
            cryptonight::aes_sbox(x0), cryptonight::aes_sbox(x1),
            cryptonight::aes_sbox(x2), cryptonight::aes_sbox(x3)) ^ aw;

        reinterpret_cast<std::uint32_t*>(slot)[lane] = cw ^ bw;
        __syncwarp(mask);
        if (lane == 0U) cryptonight::variant1_mutate(slot);
        __syncwarp(mask);

        const std::uint32_t c0 = __shfl_sync(mask, cw, base);
        const std::uint32_t c1 = __shfl_sync(mask, cw, base + 1U);
        const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                    (static_cast<std::uint64_t>(c1) << 32U);
        const std::uint32_t j = static_cast<std::uint32_t>((c_low >> 4U) & address_mask);
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(j) * 16U;

        const std::uint32_t t_word = reinterpret_cast<const std::uint32_t*>(slot2)[lane];
        const std::uint32_t t0_lo = __shfl_sync(mask, t_word, base);
        const std::uint32_t t0_hi = __shfl_sync(mask, t_word, base + 1U);
        const std::uint64_t t0 = static_cast<std::uint64_t>(t0_lo) |
                                 (static_cast<std::uint64_t>(t0_hi) << 32U);

        const std::uint32_t old_a2 = __shfl_sync(mask, aw, base + 2U);
        const std::uint32_t old_a3 = __shfl_sync(mask, aw, base + 3U);

        std::uint32_t na0_lo = 0U;
        std::uint32_t na0_hi = 0U;
        std::uint32_t na1_lo = 0U;
        std::uint32_t na1_hi = 0U;
        if (lane == 0U) {
            const std::uint64_t lo = c_low * t0;
            const std::uint64_t hi = __umul64hi(c_low, t0);
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

        std::uint32_t store_word = 0U;
        if (lane == 0U) store_word = na0_lo;
        else if (lane == 1U) store_word = na0_hi;
        else if (lane == 2U) store_word = na1_lo ^ tweak_lo;
        else store_word = na1_hi ^ tweak_hi;
        reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

        const std::uint32_t na_word = lane == 0U ? na0_lo :
                                      lane == 1U ? na0_hi :
                                      lane == 2U ? na1_lo : na1_hi;
        aw = na_word ^ t_word;
        bw = cw;
        __syncwarp(mask);
    }
}

template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_final_stage_cooperative8(std::uint8_t* states,
                                                      std::size_t count,
                                                      std::uint8_t* scratchpads,
                                                      cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int lane = warp_lane & 7U;
    const unsigned int base = warp_lane & ~7U;
    const unsigned int group = global_thread >> 3U;
    if (group >= count) return;
    const unsigned int mask = 0xFFU << base;

    std::uint8_t* state = states + static_cast<std::size_t>(group) * kStateBytes;
    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    auto& ctx = contexts[group];

    alignas(16) std::uint8_t text[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        text[i] = ctx.state[64 + static_cast<int>(lane) * 16 + i];

    alignas(16) std::uint8_t expanded[240];
    cryptonight::aes256_expand_key(ctx.state + 32, expanded);

    #pragma unroll 1
    for (std::size_t round = 0; round < init_rounds; ++round) {
        const std::uint8_t* in = scratch + round * 128U + static_cast<std::size_t>(lane) * 16U;
        #pragma unroll
        for (int i = 0; i < 16; ++i) text[i] ^= in[i];

        if constexpr (UseTTable)
            cryptonight::aes_pseudo_round_ttable(text, expanded, aes_tables);
        else
            cryptonight::aes_pseudo_round(text, expanded);
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
        ctx.state[64 + static_cast<int>(lane) * 16 + i] = text[i];
    __syncwarp(mask);

    if (lane == 0U) {
        cryptonight::keccak_permute_state(ctx.state);
        std::uint8_t digest[32];
        cryptonight::dispatch_extra_hash(static_cast<std::uint8_t>(ctx.state[0] & 3U), ctx.state, digest);
        #pragma unroll
        for (int i = 0; i < 32; ++i) state[i] = digest[i];
        #pragma unroll
        for (int i = 32; i < 64; ++i) state[i] = 0;
    }
}

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant_production(cudaStream_t stream,
                                                  std::uint8_t* states,
                                                  std::size_t count,
                                                  std::uint8_t* scratchpads,
                                                  cryptonight::SplitContext* contexts,
                                                  const CnGeometry& geometry)
{
    int device_id = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice cooperative CN failed");
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties cooperative CN failed");
    const int warp = std::max(32, props.warpSize);

    // Eight lanes per hash for the eight independent 16-byte AES chains.
    int phase_threads = std::min(props.maxThreadsPerBlock, 128);
    phase_threads = (phase_threads / warp) * warp;
    phase_threads = std::max(warp, phase_threads);
    const std::size_t phase_total_threads = count * 8U;
    const int phase_blocks = static_cast<int>((phase_total_threads + static_cast<std::size_t>(phase_threads) - 1U) /
                                              static_cast<std::size_t>(phase_threads));

    if (geometry.aes_backend == 1)
        cryptonight_setup_stage_cooperative8<VariantIndex, true><<<phase_blocks, phase_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage_cooperative8<VariantIndex, false><<<phase_blocks, phase_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative8 setup launch failed");

    int cooperative_threads = std::max(warp, geometry.loop_threads * 4);
    cooperative_threads = std::min(cooperative_threads, props.maxThreadsPerBlock);
    cooperative_threads = (cooperative_threads / warp) * warp;
    cooperative_threads = std::max(warp, cooperative_threads);
    const std::size_t total_threads = count * 4U;
    const int cooperative_blocks = static_cast<int>((total_threads + static_cast<std::size_t>(cooperative_threads) - 1U) /
                                                    static_cast<std::size_t>(cooperative_threads));
    cryptonight_loop_stage_cooperative<VariantIndex><<<cooperative_blocks, cooperative_threads, 0, stream>>>(
        count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative loop launch failed");

    if (geometry.aes_backend == 1)
        cryptonight_final_stage_cooperative8<VariantIndex, true><<<phase_blocks, phase_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    else
        cryptonight_final_stage_cooperative8<VariantIndex, false><<<phase_blocks, phase_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative8 final launch failed");
}

} // anonymous namespace
