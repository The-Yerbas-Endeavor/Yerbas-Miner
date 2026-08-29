// Production CryptoNight phase-2 selector.
//
// The shared-memory T-table/coalesced four-lane kernel is the promoted baseline.
// Experimental competitors are architecture-generic and are selected only after
// exact scratchpad parity plus a measured >=2% win on the current GPU/variant.
//
// Modes:
//   4   historical cooperative reference
//   44  promoted four-lane T-table/coalesced baseline
//   222 two-lane T-table + 64-byte shared tile
//   444 four-lane T-table/coalesced + cache-global (.cg) scratchpad traffic

namespace {

constexpr int kCnPhase2SelectorCacheRevision = 2;
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_residency_mode{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_residency_threads{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_ttable4_parity{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_tile64_parity{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_cg_parity{};

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
int cn_tile64_block_size(const cudaDeviceProp& props)
{
    constexpr std::array<int, 5> candidates{{32, 64, 128, 256, 512}};
    int best_threads = 32;
    int best_hashes_per_sm = 0;
    for (const int threads : candidates) {
        if (threads > props.maxThreadsPerBlock) continue;
        int active_blocks = 0;
        const std::size_t dynamic_bytes = cn_tile64_dynamic_shared_bytes(threads);
        const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            cryptonight_loop_stage_ttable2_tile64<VariantIndex>,
            threads,
            dynamic_bytes);
        if (rc != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        const int hashes_per_sm = active_blocks * (threads / 2);
        if (hashes_per_sm > best_hashes_per_sm ||
            (hashes_per_sm == best_hashes_per_sm && threads < best_threads)) {
            best_hashes_per_sm = hashes_per_sm;
            best_threads = threads;
        }
    }
    return best_threads;
}

inline const char* cn_residency_mode_name(int mode)
{
    if (mode == 444) return "4-lane-ttable-cg";
    if (mode == 222) return "2-lane-tile64";
    if (mode == 44) return "4-lane-ttable";
    return "4-lane-ref";
}

template <std::uint8_t VariantIndex>
std::filesystem::path cn_phase2_selector_cache_path(int device_id,
                                                    const cudaDeviceProp& props,
                                                    std::size_t batch_size)
{
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, batch_size, driver_version, runtime_version);
    return cn_cache_directory() /
        ("cn-phase2-rev" + std::to_string(kCnPhase2SelectorCacheRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_phase2_selector_cache(int device_id,
                                   const cudaDeviceProp& props,
                                   std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;

    const auto path = cn_phase2_selector_cache_path<VariantIndex>(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    unsigned int variant = 0;
    int mode = 0;
    int threads = 0;
    const int warp = std::max(32, props.warpSize);
    if (!(in >> magic >> revision >> variant >> mode >> threads) ||
        magic != "YERBAS_CN_PHASE2" || revision != kCnPhase2SelectorCacheRevision ||
        variant != static_cast<unsigned int>(VariantIndex) ||
        (mode != 4 && mode != 44 && mode != 222 && mode != 444) ||
        threads < warp || threads > props.maxThreadsPerBlock || (threads % warp) != 0) {
        return false;
    }

    if (mode == 222) {
        int active_blocks = 0;
        const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            cryptonight_loop_stage_ttable2_tile64<VariantIndex>,
            threads,
            cn_tile64_dynamic_shared_bytes(threads));
        if (rc != cudaSuccess || active_blocks <= 0) {
            cudaGetLastError();
            return false;
        }
    }

    g_cn_residency_mode[device_id][VariantIndex] = mode;
    g_cn_residency_threads[device_id][VariantIndex] = threads;
    g_cn_ttable4_parity[device_id][VariantIndex] = mode == 4 ? -1 : 1;
    g_cn_tile64_parity[device_id][VariantIndex] = mode == 222 ? 1 : 0;
    g_cn_cg_parity[device_id][VariantIndex] = mode == 444 ? 1 : 0;

    std::cout << "[GPU " << device_id << "] CryptoNight phase2 cache loaded | "
              << cryptonight::config_value(VariantIndex).name
              << " | production=" << cn_residency_mode_name(mode)
              << " | threads=" << threads << '\n';
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_phase2_selector_cache(int device_id,
                                   const cudaDeviceProp& props,
                                   std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return;
    const int mode = g_cn_residency_mode[device_id][VariantIndex];
    const int threads = g_cn_residency_threads[device_id][VariantIndex];
    if ((mode != 4 && mode != 44 && mode != 222 && mode != 444) || threads <= 0) return;

    try {
        const auto path = cn_phase2_selector_cache_path<VariantIndex>(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_PHASE2 " << kCnPhase2SelectorCacheRevision << ' '
            << static_cast<unsigned int>(VariantIndex) << ' '
            << mode << ' ' << threads << '\n';
        out.close();
        if (!out) return;

        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
        if (!ec)
            std::cout << "[GPU " << device_id << "] CryptoNight phase2 cache saved | "
                      << cryptonight::config_value(VariantIndex).name
                      << " | production=" << cn_residency_mode_name(mode)
                      << " | threads=" << threads << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] CryptoNight phase2 cache write skipped: "
                  << e.what() << '\n';
    }
}

template <std::uint8_t VariantIndex>
void cn_launch_mode(cudaStream_t stream,
                    int mode,
                    int threads,
                    std::size_t count,
                    std::uint8_t* scratchpads,
                    const cryptonight::SplitContext* contexts)
{
    const int lanes = mode == 222 ? 2 : 4;
    const std::size_t total_threads = count * static_cast<std::size_t>(lanes);
    const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));

    if (mode == 222) {
        cryptonight_loop_stage_ttable2_tile64<VariantIndex><<<
            blocks, threads, cn_tile64_dynamic_shared_bytes(threads), stream>>>(
                count, scratchpads, contexts);
    } else if (mode == 444) {
        cryptonight_loop_stage_ttable4_cg<VariantIndex><<<blocks, threads, 0, stream>>>(
            count, scratchpads, contexts);
    } else if (mode == 44) {
        cryptonight_loop_stage_ttable4_coalesced<VariantIndex><<<blocks, threads, 0, stream>>>(
            count, scratchpads, contexts);
    } else {
        cryptonight_loop_stage_cooperative<VariantIndex><<<blocks, threads, 0, stream>>>(
            count, scratchpads, contexts);
    }
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN selected loop launch failed");
}

template <std::uint8_t VariantIndex>
float cn_time_mode(cudaStream_t stream,
                   int mode,
                   int threads,
                   std::size_t count,
                   std::uint8_t* scratchpads,
                   const cryptonight::SplitContext* contexts)
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
bool cn_validate_pair(cudaStream_t stream,
                      int reference_mode,
                      int reference_threads,
                      int candidate_mode,
                      int candidate_threads,
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

    cn_launch_mode<VariantIndex>(stream, reference_mode, reference_threads, 1U, ref, ref_ctx);
    cn_launch_mode<VariantIndex>(stream, candidate_mode, candidate_threads, 1U, candidate, candidate_ctx);

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
void cn_residency_autoselect_once(cudaStream_t stream,
                                  int device_id,
                                  std::size_t count,
                                  std::uint8_t* scratchpads,
                                  const cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return;
    if (g_cn_residency_mode[device_id][VariantIndex] != 0) return;

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN selector failed");
    if (load_cn_phase2_selector_cache<VariantIndex>(device_id, props, count)) return;

    const int warp = std::max(32, props.warpSize);
    const int threads_ref = cn_occupancy_block_size(
        cryptonight_loop_stage_cooperative<VariantIndex>, warp, props.maxThreadsPerBlock);
    const int threads_t4 = cn_occupancy_block_size(
        cryptonight_loop_stage_ttable4_coalesced<VariantIndex>, warp, props.maxThreadsPerBlock);
    const int threads_cg = cn_occupancy_block_size(
        cryptonight_loop_stage_ttable4_cg<VariantIndex>, warp, props.maxThreadsPerBlock);
    const int threads_tile = cn_tile64_block_size<VariantIndex>(props);

    const bool parity_t4 = cn_validate_pair<VariantIndex>(
        stream, 4, 32, 44, 32, scratchpads, contexts);
    g_cn_ttable4_parity[device_id][VariantIndex] = parity_t4 ? 1 : -1;
    if (!parity_t4) {
        g_cn_residency_mode[device_id][VariantIndex] = 4;
        g_cn_residency_threads[device_id][VariantIndex] = threads_ref;
        save_cn_phase2_selector_cache<VariantIndex>(device_id, props, count);
        std::cout << "[CUDA CN selector] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | baseline-parity=FAIL | production=4-lane-ref\n";
        return;
    }

    const bool parity_tile = cn_validate_pair<VariantIndex>(
        stream, 44, 32, 222, 32, scratchpads, contexts);
    const bool parity_cg = cn_validate_pair<VariantIndex>(
        stream, 44, 32, 444, 32, scratchpads, contexts);
    g_cn_tile64_parity[device_id][VariantIndex] = parity_tile ? 1 : -1;
    g_cn_cg_parity[device_id][VariantIndex] = parity_cg ? 1 : -1;

    std::size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo CN selector failed");
    (void)total_bytes;
    const std::size_t by_memory = cfg.page_size ? (free_bytes / 20U) / cfg.page_size : 0U;
    const std::size_t sample_count = std::max<std::size_t>(
        1U, std::min<std::size_t>(count, std::min<std::size_t>(32U, by_memory)));

    std::uint8_t* clone = nullptr;
    cryptonight::SplitContext* clone_ctx = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone), sample_count * cfg.page_size),
               "cudaMalloc CN selector scratch failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone_ctx),
                          sample_count * sizeof(cryptonight::SplitContext)),
               "cudaMalloc CN selector ctx failed");

    auto reset = [&]() {
        check_cuda(cudaMemcpyAsync(clone, scratchpads, sample_count * cfg.page_size,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN selector scratch failed");
        check_cuda(cudaMemcpyAsync(clone_ctx, contexts,
                                   sample_count * sizeof(cryptonight::SplitContext),
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN selector ctx failed");
    };

    reset();
    const float baseline_ms = cn_time_mode<VariantIndex>(
        stream, 44, threads_t4, sample_count, clone, clone_ctx);

    float best_ms = baseline_ms;
    int best_mode = 44;
    int best_threads = threads_t4;
    float tile_ms = -1.0F;
    float cg_ms = -1.0F;

    if (parity_tile) {
        reset();
        tile_ms = cn_time_mode<VariantIndex>(
            stream, 222, threads_tile, sample_count, clone, clone_ctx);
        if (tile_ms < best_ms * 0.98F) {
            best_ms = tile_ms;
            best_mode = 222;
            best_threads = threads_tile;
        }
    }

    if (parity_cg) {
        reset();
        cg_ms = cn_time_mode<VariantIndex>(
            stream, 444, threads_cg, sample_count, clone, clone_ctx);
        if (cg_ms < best_ms * 0.98F) {
            best_ms = cg_ms;
            best_mode = 444;
            best_threads = threads_cg;
        }
    }

    cudaFree(clone_ctx);
    cudaFree(clone);

    g_cn_residency_mode[device_id][VariantIndex] = best_mode;
    g_cn_residency_threads[device_id][VariantIndex] = best_threads;
    save_cn_phase2_selector_cache<VariantIndex>(device_id, props, count);

    int tile_active_blocks = 0;
    const cudaError_t occ_rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &tile_active_blocks,
        cryptonight_loop_stage_ttable2_tile64<VariantIndex>,
        threads_tile,
        cn_tile64_dynamic_shared_bytes(threads_tile));
    if (occ_rc != cudaSuccess) {
        cudaGetLastError();
        tile_active_blocks = 0;
    }

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN selector] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | sample=" << sample_count
              << " | baseline=" << baseline_ms << " ms@" << threads_t4
              << " parity=PASS"
              << " | tile64=" << tile_ms << " ms@" << threads_tile
              << " parity=" << (parity_tile ? "PASS" : "FAIL")
              << " active-blocks/SM=" << tile_active_blocks
              << " hashes/SM=" << (tile_active_blocks * (threads_tile / 2))
              << " | cg=" << cg_ms << " ms@" << threads_cg
              << " parity=" << (parity_cg ? "PASS" : "FAIL")
              << " | production=" << cn_residency_mode_name(best_mode)
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
    (void)states;
    (void)geometry;
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN selector gate failed");

    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) {
        cn_launch_mode<VariantIndex>(stream, 4, 128, count, scratchpads, contexts);
        return;
    }

    cn_residency_autoselect_once<VariantIndex>(
        stream, device_id, count, scratchpads, contexts);

    int mode = g_cn_residency_mode[device_id][VariantIndex];
    int threads = g_cn_residency_threads[device_id][VariantIndex];
    if (mode == 0) {
        mode = 4;
        threads = 128;
    }
    cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
}

inline int cn_residency_mode(int device_id, std::uint8_t variant)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices || variant >= 6U) return 4;
    const int mode = g_cn_residency_mode[device_id][variant];
    return mode == 0 ? 4 : mode;
}

} // anonymous namespace
