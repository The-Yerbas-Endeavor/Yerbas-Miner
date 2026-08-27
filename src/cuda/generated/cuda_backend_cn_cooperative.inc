// Production cooperative CryptoNight loop.
//
// CryptoNight v1 is dependency-heavy. A one-thread-per-hash loop leaves a
// single CUDA lane carrying the complete 128-bit AES/multiply chain. This
// implementation distributes each hash across four adjacent CUDA lanes, one
// 32-bit word per lane, and uses sub-warp shuffles only for dependent values.
// The second dependent scratchpad access is also distributed across all four
// lanes so its 128-bit load/store traffic is naturally coalesced. The 64x64
// multiply remains on one lane because it is a true serial dependency.
//
// The design is generic CUDA: no GPU model or compute-capability selection is
// used. This is intentionally one production implementation, not another
// versioned experiment family. Setup/final remain on the proven split-CN
// kernels; the whole-GhostRider parity gate validates the complete pipeline.

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

        // Four adjacent lanes issue one naturally coalesced 128-bit read.
        // lane0/1 form t0, lane2/3 form t1.
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

        // Four adjacent lanes issue one naturally coalesced 128-bit write.
        // The variant tweak applies only to the stored high 64-bit half.
        std::uint32_t store_word = 0U;
        if (lane == 0U) store_word = na0_lo;
        else if (lane == 1U) store_word = na0_hi;
        else if (lane == 2U) store_word = na1_lo ^ tweak_lo;
        else store_word = na1_hi ^ tweak_hi;
        reinterpret_cast<std::uint32_t*>(slot2)[lane] = store_word;

        // State update uses the untweaked na0/na1 values XOR the original t0/t1.
        const std::uint32_t na_word = lane == 0U ? na0_lo :
                                      lane == 1U ? na0_hi :
                                      lane == 2U ? na1_lo : na1_hi;
        aw = na_word ^ t_word;
        bw = cw;
        __syncwarp(mask);
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
    const int setup_threads = geometry.setup_threads;
    const int final_threads = geometry.final_threads;
    const int setup_blocks = static_cast<int>((count + static_cast<std::size_t>(setup_threads) - 1U) /
                                              static_cast<std::size_t>(setup_threads));
    const int final_blocks = static_cast<int>((count + static_cast<std::size_t>(final_threads) - 1U) /
                                              static_cast<std::size_t>(final_threads));

    if (geometry.aes_backend == 1)
        cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative setup launch failed");

    int device_id = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice cooperative CN failed");
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties cooperative CN failed");
    const int warp = std::max(4, props.warpSize);
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
        cryptonight_final_stage<VariantIndex, true><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_final_stage<VariantIndex, false><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative final launch failed");
}

} // anonymous namespace
