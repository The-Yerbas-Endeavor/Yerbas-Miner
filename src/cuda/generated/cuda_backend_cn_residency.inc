// Occupancy-aware production selector for the parity-safe CryptoNight loops.
//
// The two-lane candidate is never used unless cuda_backend_cn_2lane.inc has
// already proven exact finalized-state parity for this GPU/variant. After
// parity is known, a small clone of the real post-setup scratchpad is used to
// benchmark the two-lane and four-lane kernels with CUDA's kernel-specific
// occupancy recommendation. Production keeps the faster mode with a 2% margin
// so measurement noise cannot promote an effectively tied candidate.

namespace {

// 0 = not benchmarked, 2 = two-lane, 4 = four-lane.
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_residency_mode{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_residency_threads{};

template <typename Kernel>
int cn_occupancy_block_size(Kernel kernel, int warp, int max_threads)
{
    int min_grid = 0;
    int block = 0;
    const cudaError_t rc = cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel, 0, 0);
    if (rc != cudaSuccess || block <= 0) {
        cudaGetLastError();
        block = 128;
    }
    block = std::min(block, max_threads);
    block = (block / warp) * warp;
    return std::max(warp, block);
}

template <std::uint8_t VariantIndex>
float cn_time_2lane_clone(cudaStream_t stream,
                          int threads,
                          std::size_t count,
                          std::uint8_t* scratchpads,
                          const cryptonight::SplitContext* contexts)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate CN 2-lane residency start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate CN 2-lane residency stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN 2-lane residency start failed");
    const std::size_t total_threads = count * 2U;
    const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    cryptonight_loop_stage_cooperative2<VariantIndex><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN 2-lane residency launch failed");
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN 2-lane residency stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN 2-lane residency failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime CN 2-lane residency failed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
float cn_time_4lane_clone(cudaStream_t stream,
                          int threads,
                          std::size_t count,
                          std::uint8_t* scratchpads,
                          const cryptonight::SplitContext* contexts)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate CN 4-lane residency start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate CN 4-lane residency stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN 4-lane residency start failed");
    const std::size_t total_threads = count * 4U;
    const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    cryptonight_loop_stage_cooperative<VariantIndex><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN 4-lane residency launch failed");
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN 4-lane residency stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN 4-lane residency failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime CN 4-lane residency failed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
void cn_residency_autoselect_once(cudaStream_t stream,
                                  int device_id,
                                  std::size_t count,
                                  std::uint8_t* scratchpads,
                                  const cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return;
    if (g_cn_residency_mode[device_id][VariantIndex] != 0) return;

    // A failed parity gate can never promote two-lane.
    if (g_cn2lane_parity[device_id][VariantIndex] < 0) {
        g_cn_residency_mode[device_id][VariantIndex] = 4;
        return;
    }
    // Wait until the parity gate has run on a real post-setup hash.
    if (g_cn2lane_parity[device_id][VariantIndex] == 0) return;

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN residency failed");
    const int warp = std::max(32, props.warpSize);

    const int threads2 = cn_occupancy_block_size(
        cryptonight_loop_stage_cooperative2<VariantIndex>, warp, props.maxThreadsPerBlock);
    const int threads4 = cn_occupancy_block_size(
        cryptonight_loop_stage_cooperative<VariantIndex>, warp, props.maxThreadsPerBlock);

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo CN residency failed");
    (void)total_bytes;

    // Keep the one-time benchmark small and safe on lower-memory GPUs. At most
    // sixteen hashes are cloned; no more than 5% of currently free VRAM is used.
    const std::size_t budget = free_bytes / 20U;
    const std::size_t by_memory = cfg.page_size == 0 ? 0U : budget / cfg.page_size;
    const std::size_t sample_count = std::min<std::size_t>(count, std::min<std::size_t>(16U, by_memory));
    if (sample_count == 0U) {
        g_cn_residency_mode[device_id][VariantIndex] = 4;
        g_cn_residency_threads[device_id][VariantIndex] = threads4;
        return;
    }

    std::uint8_t* clone_scratch = nullptr;
    cryptonight::SplitContext* clone_contexts = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone_scratch), sample_count * cfg.page_size),
               "cudaMalloc CN residency scratch clone failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone_contexts), sample_count * sizeof(cryptonight::SplitContext)),
               "cudaMalloc CN residency context clone failed");

    auto reset_clone = [&]() {
        check_cuda(cudaMemcpyAsync(clone_scratch, scratchpads, sample_count * cfg.page_size,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN residency scratch clone failed");
        check_cuda(cudaMemcpyAsync(clone_contexts, contexts, sample_count * sizeof(cryptonight::SplitContext),
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN residency context clone failed");
    };

    reset_clone();
    const float ms2 = cn_time_2lane_clone<VariantIndex>(stream, threads2, sample_count,
                                                        clone_scratch, clone_contexts);
    reset_clone();
    const float ms4 = cn_time_4lane_clone<VariantIndex>(stream, threads4, sample_count,
                                                        clone_scratch, clone_contexts);

    cudaFree(clone_contexts);
    cudaFree(clone_scratch);

    // Require at least 2% improvement to switch away from the established path.
    const bool choose2 = ms2 < (ms4 * 0.98F);
    g_cn_residency_mode[device_id][VariantIndex] = choose2 ? 2 : 4;
    g_cn_residency_threads[device_id][VariantIndex] = choose2 ? threads2 : threads4;

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN residency] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | sample=" << sample_count
              << " | 2-lane=" << ms2 << " ms@" << threads2
              << " | 4-lane=" << ms4 << " ms@" << threads4
              << " | production=" << (choose2 ? "2-lane" : "4-lane")
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex, bool UseTTable>
void launch_cn_loop_residency_gated(cudaStream_t stream,
                                    std::uint8_t* states,
                                    std::size_t count,
                                    std::uint8_t* scratchpads,
                                    cryptonight::SplitContext* contexts,
                                    const CnGeometry& geometry)
{
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN residency gate failed");

    // First use performs the existing exact finalized-state parity gate and
    // processes the live batch through the parity-selected path.
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices ||
        g_cn2lane_parity[device_id][VariantIndex] == 0) {
        launch_cn_loop_parity_gated<VariantIndex, UseTTable>(
            stream, states, count, scratchpads, contexts, geometry);
        return;
    }

    cn_residency_autoselect_once<VariantIndex>(stream, device_id, count, scratchpads, contexts);
    const int mode = g_cn_residency_mode[device_id][VariantIndex];
    const int threads = g_cn_residency_threads[device_id][VariantIndex];

    if (mode == 2) {
        const std::size_t total_threads = count * 2U;
        const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        cryptonight_loop_stage_cooperative2<VariantIndex><<<blocks, threads, 0, stream>>>(
            count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "GhostRider CUDA CN residency-selected 2-lane launch failed");
    } else {
        const int use_threads = threads > 0 ? threads : 128;
        const std::size_t total_threads = count * 4U;
        const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(use_threads) - 1U) /
                                            static_cast<std::size_t>(use_threads));
        cryptonight_loop_stage_cooperative<VariantIndex><<<blocks, use_threads, 0, stream>>>(
            count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "GhostRider CUDA CN residency-selected 4-lane launch failed");
    }
}

inline int cn_residency_mode(int device_id, std::uint8_t variant)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices || variant >= 6U) return 4;
    const int mode = g_cn_residency_mode[device_id][variant];
    if (mode == 2 || mode == 4) return mode;
    return g_cn2lane_parity[device_id][variant] == 1 ? 2 : 4;
}

} // anonymous namespace
