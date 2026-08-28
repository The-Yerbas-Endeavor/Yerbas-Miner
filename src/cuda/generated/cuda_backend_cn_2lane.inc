// Two-lane CryptoNight memory-loop candidate with direct production parity gate.
//
// The existing four-lane loop remains the reference implementation. On first use
// of each GPU/variant pair, one real post-setup hash is cloned twice, run through
// the reference and candidate loops independently, finalized through the same
// production final kernel, and compared. The two-lane path is promoted only
// after exact output parity for that GPU/variant.

namespace {

constexpr int kCn2LaneMaxDevices = 32;
// 0 = unknown, 1 = pass/promoted, -1 = failed/use four-lane reference.
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn2lane_parity{};

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_cooperative2(std::size_t count,
                                                    std::uint8_t* scratchpads,
                                                    const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;

    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 1U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~1U;
    const unsigned int group = global_thread >> 1U;
    if (group >= count) return;
    const unsigned int mask = 0x3U << base;

    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];

    // One lane owns each 64-bit half of A/B. This cuts the cooperation width in
    // half while keeping the full 128-bit dependency state resident in registers.
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

        // lane 0 computes AES output words 0/1; lane 1 computes words 2/3.
        // The byte pattern is identical relative to each lane's own 64-bit half.
        const std::uint32_t cw_lo = cn_coop_mix_column(
            cryptonight::aes_sbox(sb[0]), cryptonight::aes_sbox(sb[5]),
            cryptonight::aes_sbox(ob[2]), cryptonight::aes_sbox(ob[7])) ^
            static_cast<std::uint32_t>(a);
        const std::uint32_t cw_hi = cn_coop_mix_column(
            cryptonight::aes_sbox(sb[4]), cryptonight::aes_sbox(ob[1]),
            cryptonight::aes_sbox(ob[6]), cryptonight::aes_sbox(sb[3])) ^
            static_cast<std::uint32_t>(a >> 32U);
        const std::uint64_t c = static_cast<std::uint64_t>(cw_lo) |
                                (static_cast<std::uint64_t>(cw_hi) << 32U);

        reinterpret_cast<std::uint64_t*>(slot)[lane] = c ^ b;
        __syncwarp(mask);
        if (lane == 0U) cryptonight::variant1_mutate(slot);
        __syncwarp(mask);

        const std::uint64_t c0 = __shfl_sync(mask, c, base);
        const std::uint32_t j = static_cast<std::uint32_t>((c0 >> 4U) & address_mask);
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(j) * 16U;
        const std::uint64_t t = reinterpret_cast<const std::uint64_t*>(slot2)[lane];
        const std::uint64_t t0 = __shfl_sync(mask, t, base);
        const std::uint64_t old_a1 = __shfl_sync(mask, a, base + 1U);

        std::uint64_t next0 = 0U;
        std::uint64_t next1 = 0U;
        if (lane == 0U) {
            const std::uint64_t lo = c0 * t0;
            const std::uint64_t hi = __umul64hi(c0, t0);
            next0 = a0 + hi;
            next1 = old_a1 + lo;
        }
        next0 = __shfl_sync(mask, next0, base);
        next1 = __shfl_sync(mask, next1, base);

        reinterpret_cast<std::uint64_t*>(slot2)[lane] = lane == 0U ? next0 : (next1 ^ tweak);
        a = (lane == 0U ? next0 : next1) ^ t;
        b = c;
        __syncwarp(mask);
    }
}

template <std::uint8_t VariantIndex>
void launch_cn_loop_2lane(cudaStream_t stream,
                          std::size_t count,
                          std::uint8_t* scratchpads,
                          const cryptonight::SplitContext* contexts,
                          const CnGeometry& geometry)
{
    int device_id = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN 2-lane failed");
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN 2-lane failed");
    const int warp = std::max(32, props.warpSize);
    int threads = std::max(warp, geometry.loop_threads * 2);
    threads = std::min(threads, props.maxThreadsPerBlock);
    threads = (threads / warp) * warp;
    threads = std::max(warp, threads);
    const std::size_t total_threads = count * 2U;
    const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    cryptonight_loop_stage_cooperative2<VariantIndex><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight 2-lane loop launch failed");
}

template <std::uint8_t VariantIndex>
void launch_cn_loop_4lane_reference(cudaStream_t stream,
                                    std::size_t count,
                                    std::uint8_t* scratchpads,
                                    const cryptonight::SplitContext* contexts,
                                    const CnGeometry& geometry)
{
    int device_id = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN 4-lane reference failed");
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN 4-lane reference failed");
    const int warp = std::max(32, props.warpSize);
    int threads = std::max(warp, geometry.loop_threads * 4);
    threads = std::min(threads, props.maxThreadsPerBlock);
    threads = (threads / warp) * warp;
    threads = std::max(warp, threads);
    const std::size_t total_threads = count * 4U;
    const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    cryptonight_loop_stage_cooperative<VariantIndex><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight 4-lane reference launch failed");
}

template <std::uint8_t VariantIndex, bool UseTTable>
bool cn_2lane_parity_once(cudaStream_t stream,
                          const std::uint8_t* states,
                          const std::uint8_t* scratchpads,
                          const cryptonight::SplitContext* contexts,
                          const CnGeometry& geometry)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t *candidate_state = nullptr, *reference_state = nullptr;
    std::uint8_t *candidate_scratch = nullptr, *reference_scratch = nullptr;
    cryptonight::SplitContext *candidate_ctx = nullptr, *reference_ctx = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate_state), kStateBytes), "cudaMalloc CN 2-lane parity candidate state failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&reference_state), kStateBytes), "cudaMalloc CN 2-lane parity reference state failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate_scratch), cfg.page_size), "cudaMalloc CN 2-lane parity candidate scratch failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&reference_scratch), cfg.page_size), "cudaMalloc CN 2-lane parity reference scratch failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate_ctx), sizeof(cryptonight::SplitContext)), "cudaMalloc CN 2-lane parity candidate context failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&reference_ctx), sizeof(cryptonight::SplitContext)), "cudaMalloc CN 2-lane parity reference context failed");

    check_cuda(cudaMemcpyAsync(candidate_state, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 2-lane candidate state failed");
    check_cuda(cudaMemcpyAsync(reference_state, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 2-lane reference state failed");
    check_cuda(cudaMemcpyAsync(candidate_scratch, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 2-lane candidate scratch failed");
    check_cuda(cudaMemcpyAsync(reference_scratch, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 2-lane reference scratch failed");
    check_cuda(cudaMemcpyAsync(candidate_ctx, contexts, sizeof(cryptonight::SplitContext), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 2-lane candidate context failed");
    check_cuda(cudaMemcpyAsync(reference_ctx, contexts, sizeof(cryptonight::SplitContext), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 2-lane reference context failed");

    launch_cn_loop_2lane<VariantIndex>(stream, 1U, candidate_scratch, candidate_ctx, geometry);
    launch_cn_loop_4lane_reference<VariantIndex>(stream, 1U, reference_scratch, reference_ctx, geometry);

    cryptonight_final_stage_cooperative8_sharedkey<VariantIndex, UseTTable><<<1, 32, 0, stream>>>(
        candidate_state, 1U, candidate_scratch, candidate_ctx);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN 2-lane parity candidate final failed");
    cryptonight_final_stage_cooperative8_sharedkey<VariantIndex, UseTTable><<<1, 32, 0, stream>>>(
        reference_state, 1U, reference_scratch, reference_ctx);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN 2-lane parity reference final failed");

    std::array<std::uint8_t, kStateBytes> candidate{};
    std::array<std::uint8_t, kStateBytes> reference{};
    check_cuda(cudaMemcpyAsync(candidate.data(), candidate_state, kStateBytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpy CN 2-lane candidate result failed");
    check_cuda(cudaMemcpyAsync(reference.data(), reference_state, kStateBytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpy CN 2-lane reference result failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CN 2-lane parity failed");

    cudaFree(reference_ctx);
    cudaFree(candidate_ctx);
    cudaFree(reference_scratch);
    cudaFree(candidate_scratch);
    cudaFree(reference_state);
    cudaFree(candidate_state);
    return candidate == reference;
}

template <std::uint8_t VariantIndex, bool UseTTable>
void launch_cn_loop_parity_gated(cudaStream_t stream,
                                 std::uint8_t* states,
                                 std::size_t count,
                                 std::uint8_t* scratchpads,
                                 cryptonight::SplitContext* contexts,
                                 const CnGeometry& geometry)
{
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN 2-lane gate failed");

    int status = -1;
    int* cached = nullptr;
    if (device_id >= 0 && device_id < kCn2LaneMaxDevices) {
        cached = &g_cn2lane_parity[device_id][VariantIndex];
        status = *cached;
    }

    if (cached != nullptr && status == 0) {
        const bool pass = cn_2lane_parity_once<VariantIndex, UseTTable>(
            stream, states, scratchpads, contexts, geometry);
        *cached = pass ? 1 : -1;
        status = *cached;
        std::cout << "[CUDA CN 2-lane parity] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | result=" << (pass ? "PASS" : "FAIL")
                  << " | production=" << (pass ? "2-lane" : "4-lane") << '\n';
    }

    if (status == 1)
        launch_cn_loop_2lane<VariantIndex>(stream, count, scratchpads, contexts, geometry);
    else
        launch_cn_loop_4lane_reference<VariantIndex>(stream, count, scratchpads, contexts, geometry);
}

} // anonymous namespace
