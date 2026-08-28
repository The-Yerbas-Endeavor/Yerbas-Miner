// Production CryptoNight loop selector.
//
// The original four-lane loop remains the correctness reference. After the
// existing two-lane finalized-state gate succeeds, the selector validates the
// new shared-memory AES T-table loops against the reference scratchpad result,
// benchmarks all parity-safe candidates with kernel-specific occupancy launch
// sizes, then keeps the fastest path for each GPU/variant.

namespace {

// 0 unknown, 4 original four-lane, 22 T-table two-lane, 44 T-table four-lane.
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_residency_mode{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_residency_threads{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_ttable2_parity{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_ttable4_parity{};

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

inline const char* cn_residency_mode_name(int mode)
{
    if (mode == 22) return "2-lane-ttable";
    if (mode == 44) return "4-lane-ttable";
    return "4-lane-ref";
}

template <std::uint8_t VariantIndex>
void cn_launch_mode(cudaStream_t stream, int mode, int threads, std::size_t count,
                    std::uint8_t* scratchpads, const cryptonight::SplitContext* contexts)
{
    const int lanes = mode == 22 ? 2 : 4;
    const std::size_t total_threads = count * static_cast<std::size_t>(lanes);
    const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    if (mode == 22)
        cryptonight_loop_stage_ttable2<VariantIndex><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    else if (mode == 44)
        cryptonight_loop_stage_ttable4_coalesced<VariantIndex><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    else
        cryptonight_loop_stage_cooperative<VariantIndex><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN selected loop launch failed");
}

template <std::uint8_t VariantIndex>
float cn_time_mode(cudaStream_t stream, int mode, int threads, std::size_t count,
                   std::uint8_t* scratchpads, const cryptonight::SplitContext* contexts)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate CN loop test start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate CN loop test stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN loop test start failed");
    cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN loop test stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN loop test failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime CN loop test failed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
bool cn_validate_mode(cudaStream_t stream, int mode, int candidate_threads,
                      const std::uint8_t* scratchpads,
                      const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* ref = nullptr;
    std::uint8_t* candidate = nullptr;
    cryptonight::SplitContext* ref_ctx = nullptr;
    cryptonight::SplitContext* candidate_ctx = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ref), cfg.page_size), "cudaMalloc CN parity ref failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate), cfg.page_size), "cudaMalloc CN parity candidate failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ref_ctx), sizeof(cryptonight::SplitContext)), "cudaMalloc CN parity ref ctx failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate_ctx), sizeof(cryptonight::SplitContext)), "cudaMalloc CN parity candidate ctx failed");
    check_cuda(cudaMemcpyAsync(ref, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN parity ref failed");
    check_cuda(cudaMemcpyAsync(candidate, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN parity candidate failed");
    check_cuda(cudaMemcpyAsync(ref_ctx, contexts, sizeof(cryptonight::SplitContext), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN parity ref ctx failed");
    check_cuda(cudaMemcpyAsync(candidate_ctx, contexts, sizeof(cryptonight::SplitContext), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN parity candidate ctx failed");

    cn_launch_mode<VariantIndex>(stream, 4, 32, 1U, ref, ref_ctx);
    cn_launch_mode<VariantIndex>(stream, mode, candidate_threads, 1U, candidate, candidate_ctx);
    std::vector<std::uint8_t> host_ref(cfg.page_size), host_candidate(cfg.page_size);
    check_cuda(cudaMemcpyAsync(host_ref.data(), ref, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy CN parity ref result failed");
    check_cuda(cudaMemcpyAsync(host_candidate.data(), candidate, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy CN parity candidate result failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CN parity result failed");

    cudaFree(candidate_ctx);
    cudaFree(ref_ctx);
    cudaFree(candidate);
    cudaFree(ref);
    return host_ref == host_candidate;
}

template <std::uint8_t VariantIndex>
void cn_residency_autoselect_once(cudaStream_t stream, int device_id, std::size_t count,
                                  std::uint8_t* scratchpads,
                                  const cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return;
    if (g_cn_residency_mode[device_id][VariantIndex] != 0) return;
    if (g_cn2lane_parity[device_id][VariantIndex] <= 0) {
        g_cn_residency_mode[device_id][VariantIndex] = 4;
        g_cn_residency_threads[device_id][VariantIndex] = 128;
        return;
    }

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN selector failed");
    const int warp = std::max(32, props.warpSize);
    const int threads_ref = cn_occupancy_block_size(cryptonight_loop_stage_cooperative<VariantIndex>, warp, props.maxThreadsPerBlock);
    const int threads_t2 = cn_occupancy_block_size(cryptonight_loop_stage_ttable2<VariantIndex>, warp, props.maxThreadsPerBlock);
    const int threads_t4 = cn_occupancy_block_size(cryptonight_loop_stage_ttable4_coalesced<VariantIndex>, warp, props.maxThreadsPerBlock);

    const bool parity_t2 = cn_validate_mode<VariantIndex>(stream, 22, threads_t2, scratchpads, contexts);
    const bool parity_t4 = cn_validate_mode<VariantIndex>(stream, 44, threads_t4, scratchpads, contexts);
    g_cn_ttable2_parity[device_id][VariantIndex] = parity_t2 ? 1 : -1;
    g_cn_ttable4_parity[device_id][VariantIndex] = parity_t4 ? 1 : -1;

    std::size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo CN selector failed");
    (void)total_bytes;
    const std::size_t by_memory = cfg.page_size ? (free_bytes / 20U) / cfg.page_size : 0U;
    const std::size_t sample_count = std::max<std::size_t>(1U, std::min<std::size_t>(count, std::min<std::size_t>(32U, by_memory)));

    std::uint8_t* clone = nullptr;
    cryptonight::SplitContext* clone_ctx = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone), sample_count * cfg.page_size), "cudaMalloc CN selector scratch failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone_ctx), sample_count * sizeof(cryptonight::SplitContext)), "cudaMalloc CN selector ctx failed");
    auto reset = [&]() {
        check_cuda(cudaMemcpyAsync(clone, scratchpads, sample_count * cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN selector scratch failed");
        check_cuda(cudaMemcpyAsync(clone_ctx, contexts, sample_count * sizeof(cryptonight::SplitContext), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN selector ctx failed");
    };

    reset();
    const float ref_ms = cn_time_mode<VariantIndex>(stream, 4, threads_ref, sample_count, clone, clone_ctx);
    float best_ms = ref_ms;
    int best_mode = 4;
    int best_threads = threads_ref;
    float t2_ms = -1.0F, t4_ms = -1.0F;
    if (parity_t2) {
        reset();
        t2_ms = cn_time_mode<VariantIndex>(stream, 22, threads_t2, sample_count, clone, clone_ctx);
        if (t2_ms < best_ms * 0.98F) { best_ms = t2_ms; best_mode = 22; best_threads = threads_t2; }
    }
    if (parity_t4) {
        reset();
        t4_ms = cn_time_mode<VariantIndex>(stream, 44, threads_t4, sample_count, clone, clone_ctx);
        if (t4_ms < best_ms * 0.98F) { best_ms = t4_ms; best_mode = 44; best_threads = threads_t4; }
    }
    cudaFree(clone_ctx);
    cudaFree(clone);

    g_cn_residency_mode[device_id][VariantIndex] = best_mode;
    g_cn_residency_threads[device_id][VariantIndex] = best_threads;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN selector] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | sample=" << sample_count
              << " | ref=" << ref_ms << " ms@" << threads_ref
              << " | t2=" << t2_ms << " ms@" << threads_t2 << " parity=" << (parity_t2 ? "PASS" : "FAIL")
              << " | t4=" << t4_ms << " ms@" << threads_t4 << " parity=" << (parity_t4 ? "PASS" : "FAIL")
              << " | production=" << cn_residency_mode_name(best_mode)
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex, bool UseTTable>
void launch_cn_loop_residency_gated(cudaStream_t stream, std::uint8_t* states, std::size_t count,
                                    std::uint8_t* scratchpads,
                                    cryptonight::SplitContext* contexts,
                                    const CnGeometry& geometry)
{
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN selector gate failed");
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices || g_cn2lane_parity[device_id][VariantIndex] == 0) {
        launch_cn_loop_parity_gated<VariantIndex, UseTTable>(stream, states, count, scratchpads, contexts, geometry);
        return;
    }

    cn_residency_autoselect_once<VariantIndex>(stream, device_id, count, scratchpads, contexts);
    int mode = g_cn_residency_mode[device_id][VariantIndex];
    int threads = g_cn_residency_threads[device_id][VariantIndex];
    if (mode == 0) { mode = 4; threads = 128; }
    cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
}

inline int cn_residency_mode(int device_id, std::uint8_t variant)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices || variant >= 6U) return 4;
    const int mode = g_cn_residency_mode[device_id][variant];
    return mode == 0 ? 4 : mode;
}

} // anonymous namespace
