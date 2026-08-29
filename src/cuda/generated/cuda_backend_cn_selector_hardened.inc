// Hardened production CryptoNight phase-2 selector.
//
// This layer deliberately reuses the already parity-tested production kernels
// from cuda_backend_cn_residency.inc, but replaces one-shot timing decisions with
// repeated alternating A/B measurements. It is architecture-generic: no GPU name,
// compute capability, or generation is used to force a kernel choice.
//
// Promotion rules:
//   * exact full-scratchpad parity is mandatory;
//   * one warm-up run is discarded for each implementation;
//   * five measured passes are collected from identical cloned post-setup state;
//   * the median candidate time must beat the median T-table baseline by >=5%.

namespace {

constexpr int kCnHardenedSelectorRevision = 3;
constexpr int kCnHardenedPasses = 5;
constexpr float kCnHardenedPromotionRatio = 0.95F;
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_hardened_mode{};
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_hardened_threads{};

template <std::size_t N>
float cn_hardened_median(std::array<float, N> values)
{
    std::sort(values.begin(), values.end());
    if constexpr ((N & 1U) != 0U)
        return values[N / 2U];
    else
        return 0.5F * (values[N / 2U - 1U] + values[N / 2U]);
}

template <std::uint8_t VariantIndex>
std::filesystem::path cn_hardened_cache_path(int device_id,
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
        ("cn-phase2-hardened-rev" + std::to_string(kCnHardenedSelectorRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_hardened_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;

    const auto path = cn_hardened_cache_path<VariantIndex>(device_id, props, batch_size);
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
        magic != "YERBAS_CN_PHASE2_HARDENED" || revision != kCnHardenedSelectorRevision ||
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

    g_cn_hardened_mode[device_id][VariantIndex] = mode;
    g_cn_hardened_threads[device_id][VariantIndex] = threads;
    std::cout << "[GPU " << device_id << "] CryptoNight hardened phase2 cache loaded | "
              << cryptonight::config_value(VariantIndex).name
              << " | production=" << cn_residency_mode_name(mode)
              << " | threads=" << threads << '\n';
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_hardened_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return;
    const int mode = g_cn_hardened_mode[device_id][VariantIndex];
    const int threads = g_cn_hardened_threads[device_id][VariantIndex];
    if ((mode != 4 && mode != 44 && mode != 222 && mode != 444) || threads <= 0) return;

    try {
        const auto path = cn_hardened_cache_path<VariantIndex>(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_PHASE2_HARDENED " << kCnHardenedSelectorRevision << ' '
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
    } catch (...) {
        // Selection cache is an optimization only. Mining must remain available.
    }
}

template <std::uint8_t VariantIndex>
void cn_hardened_autoselect_once(cudaStream_t stream,
                                 int device_id,
                                 std::size_t count,
                                 std::uint8_t* scratchpads,
                                 const cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) return;
    if (g_cn_hardened_mode[device_id][VariantIndex] != 0) return;

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties hardened CN selector failed");
    if (load_cn_hardened_cache<VariantIndex>(device_id, props, count)) return;

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
    if (!parity_t4) {
        g_cn_hardened_mode[device_id][VariantIndex] = 4;
        g_cn_hardened_threads[device_id][VariantIndex] = threads_ref;
        save_cn_hardened_cache<VariantIndex>(device_id, props, count);
        std::cout << "[CUDA CN hardened selector] GPU " << device_id
                  << " | " << cfg.name
                  << " | baseline-parity=FAIL | production=4-lane-ref\n";
        return;
    }

    const bool parity_tile = cn_validate_pair<VariantIndex>(
        stream, 44, 32, 222, 32, scratchpads, contexts);
    const bool parity_cg = cn_validate_pair<VariantIndex>(
        stream, 44, 32, 444, 32, scratchpads, contexts);

    std::size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo hardened CN selector failed");
    (void)total_bytes;
    const std::size_t by_memory = cfg.page_size ? (free_bytes / 20U) / cfg.page_size : 0U;
    const std::size_t sample_count = std::max<std::size_t>(
        1U, std::min<std::size_t>(count, std::min<std::size_t>(32U, by_memory)));

    std::uint8_t* clone = nullptr;
    cryptonight::SplitContext* clone_ctx = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone), sample_count * cfg.page_size),
               "cudaMalloc hardened CN selector scratch failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone_ctx),
                          sample_count * sizeof(cryptonight::SplitContext)),
               "cudaMalloc hardened CN selector ctx failed");

    auto reset = [&]() {
        check_cuda(cudaMemcpyAsync(clone, scratchpads, sample_count * cfg.page_size,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy hardened CN selector scratch failed");
        check_cuda(cudaMemcpyAsync(clone_ctx, contexts,
                                   sample_count * sizeof(cryptonight::SplitContext),
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy hardened CN selector ctx failed");
    };
    auto timed = [&](int mode, int threads) {
        reset();
        return cn_time_mode<VariantIndex>(stream, mode, threads, sample_count, clone, clone_ctx);
    };

    // Discard one warm-up execution for each implementation that can compete.
    (void)timed(44, threads_t4);
    if (parity_tile) (void)timed(222, threads_tile);
    if (parity_cg) (void)timed(444, threads_cg);

    std::array<float, kCnHardenedPasses> baseline_times{};
    std::array<float, kCnHardenedPasses> tile_times{};
    std::array<float, kCnHardenedPasses> cg_times{};
    tile_times.fill(-1.0F);
    cg_times.fill(-1.0F);

    // Alternate ordering to reduce temperature/clock/order bias. Every timing
    // starts from an identical clone of the real post-setup scratchpad/context.
    for (int pass = 0; pass < kCnHardenedPasses; ++pass) {
        if ((pass & 1) == 0) {
            baseline_times[pass] = timed(44, threads_t4);
            if (parity_tile) tile_times[pass] = timed(222, threads_tile);
            if (parity_cg) cg_times[pass] = timed(444, threads_cg);
        } else {
            if (parity_cg) cg_times[pass] = timed(444, threads_cg);
            if (parity_tile) tile_times[pass] = timed(222, threads_tile);
            baseline_times[pass] = timed(44, threads_t4);
        }
    }

    const float baseline_ms = cn_hardened_median(baseline_times);
    const float tile_ms = parity_tile ? cn_hardened_median(tile_times) : -1.0F;
    const float cg_ms = parity_cg ? cn_hardened_median(cg_times) : -1.0F;

    int best_mode = 44;
    int best_threads = threads_t4;
    float best_ms = baseline_ms;
    if (parity_tile && tile_ms < baseline_ms * kCnHardenedPromotionRatio && tile_ms < best_ms) {
        best_mode = 222;
        best_threads = threads_tile;
        best_ms = tile_ms;
    }
    if (parity_cg && cg_ms < baseline_ms * kCnHardenedPromotionRatio && cg_ms < best_ms) {
        best_mode = 444;
        best_threads = threads_cg;
        best_ms = cg_ms;
    }

    cudaFree(clone_ctx);
    cudaFree(clone);

    g_cn_hardened_mode[device_id][VariantIndex] = best_mode;
    g_cn_hardened_threads[device_id][VariantIndex] = best_threads;
    save_cn_hardened_cache<VariantIndex>(device_id, props, count);

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN hardened selector] GPU " << device_id
              << " | " << cfg.name
              << " | sample=" << sample_count
              << " passes=" << kCnHardenedPasses
              << " threshold=5%"
              << " | baseline-median=" << baseline_ms << " ms@" << threads_t4
              << " | tile64-median=" << tile_ms << " ms@" << threads_tile
              << " parity=" << (parity_tile ? "PASS" : "FAIL")
              << " | cg-median=" << cg_ms << " ms@" << threads_cg
              << " parity=" << (parity_cg ? "PASS" : "FAIL")
              << " | production=" << cn_residency_mode_name(best_mode)
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex, bool UseTTable>
void launch_cn_loop_hardened(cudaStream_t stream,
                             std::uint8_t* states,
                             std::size_t count,
                             std::uint8_t* scratchpads,
                             cryptonight::SplitContext* contexts,
                             const CnGeometry& geometry)
{
    (void)states;
    (void)geometry;
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice hardened CN selector gate failed");
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) {
        cn_launch_mode<VariantIndex>(stream, 4, 128, count, scratchpads, contexts);
        return;
    }

    cn_hardened_autoselect_once<VariantIndex>(stream, device_id, count, scratchpads, contexts);
    int mode = g_cn_hardened_mode[device_id][VariantIndex];
    int threads = g_cn_hardened_threads[device_id][VariantIndex];
    if (mode == 0) {
        mode = 4;
        threads = 128;
    }
    cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
}

inline int cn_hardened_mode(int device_id, std::uint8_t variant)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices || variant >= 6U) return 4;
    const int mode = g_cn_hardened_mode[device_id][variant];
    return mode == 0 ? 4 : mode;
}

} // anonymous namespace
