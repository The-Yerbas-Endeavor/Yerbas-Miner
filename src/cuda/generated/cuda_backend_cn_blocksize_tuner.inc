// Runtime block-size tuner for the selected production CryptoNight phase-2 kernel.
//
// Kernel selection and exact parity remain owned by the hardened selector. This
// layer tunes only launch geometry for that already-approved kernel. CUDA's
// occupancy recommendation is a useful starting point, but it is not always the
// fastest block size for a memory-latency-heavy CryptoNight loop. Benchmarking a
// compact warp-aligned candidate set keeps this architecture-generic.

namespace {

constexpr int kCnBlockTuneRevision = 1;
constexpr int kCnBlockTunePasses = 3;
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_block_tuned_threads{};

template <std::uint8_t VariantIndex>
std::filesystem::path cn_block_tune_cache_path(int device_id,
                                               const cudaDeviceProp& props,
                                               std::size_t batch_size,
                                               int mode)
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
        ("cn-block-rev" + std::to_string(kCnBlockTuneRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) +
         "-m" + std::to_string(mode) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_block_tune_cache(int device_id,
                              const cudaDeviceProp& props,
                              std::size_t batch_size,
                              int mode)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_block_tune_cache_path<VariantIndex>(device_id, props, batch_size, mode);
    if (path.empty()) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0, variant = -1, cached_mode = 0, threads = 0;
    if (!(in >> magic >> revision >> variant >> cached_mode >> threads) ||
        magic != "YERBAS_CN_BLOCK" || revision != kCnBlockTuneRevision ||
        variant != static_cast<int>(VariantIndex) || cached_mode != mode ||
        threads < props.warpSize || threads > props.maxThreadsPerBlock ||
        (threads % props.warpSize) != 0)
        return false;
    g_cn_block_tuned_threads[device_id][VariantIndex] = threads;
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_block_tune_cache(int device_id,
                              const cudaDeviceProp& props,
                              std::size_t batch_size,
                              int mode,
                              int threads)
{
    try {
        const auto path = cn_block_tune_cache_path<VariantIndex>(device_id, props, batch_size, mode);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (out)
            out << "YERBAS_CN_BLOCK " << kCnBlockTuneRevision << ' '
                << static_cast<int>(VariantIndex) << ' ' << mode << ' ' << threads << '\n';
    } catch (...) {}
}

template <std::uint8_t VariantIndex>
int cn_tune_selected_block_size(cudaStream_t stream,
                                int device_id,
                                const cudaDeviceProp& props,
                                int mode,
                                int baseline_threads,
                                std::size_t count,
                                std::uint8_t* scratchpads,
                                const cryptonight::SplitContext* contexts)
{
    if (g_cn_block_tuned_threads[device_id][VariantIndex] > 0)
        return g_cn_block_tuned_threads[device_id][VariantIndex];
    if (load_cn_block_tune_cache<VariantIndex>(device_id, props, count, mode))
        return g_cn_block_tuned_threads[device_id][VariantIndex];

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo CN block tuner failed");
    (void)total_bytes;
    const std::size_t by_memory = cfg.page_size ? (free_bytes / 20U) / cfg.page_size : 0U;
    const std::size_t sample_count = std::max<std::size_t>(
        1U, std::min<std::size_t>(count, std::min<std::size_t>(32U, by_memory)));

    std::uint8_t* clone = nullptr;
    cryptonight::SplitContext* clone_ctx = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone), sample_count * cfg.page_size),
               "cudaMalloc CN block tuner scratch failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone_ctx),
                          sample_count * sizeof(cryptonight::SplitContext)),
               "cudaMalloc CN block tuner context failed");

    auto reset = [&]() {
        check_cuda(cudaMemcpyAsync(clone, scratchpads, sample_count * cfg.page_size,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN block tuner scratch failed");
        check_cuda(cudaMemcpyAsync(clone_ctx, contexts,
                                   sample_count * sizeof(cryptonight::SplitContext),
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN block tuner context failed");
    };

    std::vector<int> candidates;
    const auto add = [&](int threads) {
        if (threads < props.warpSize || threads > props.maxThreadsPerBlock ||
            (threads % props.warpSize) != 0) return;
        if (mode == 222) {
            int active = 0;
            const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, cryptonight_loop_stage_ttable2_tile64<VariantIndex>, threads,
                cn_tile64_dynamic_shared_bytes(threads));
            if (rc != cudaSuccess || active <= 0) { cudaGetLastError(); return; }
        }
        if (std::find(candidates.begin(), candidates.end(), threads) == candidates.end())
            candidates.push_back(threads);
    };

    add(baseline_threads);
    for (int threads = props.warpSize; threads <= std::min(256, props.maxThreadsPerBlock);
         threads += props.warpSize)
        add(threads);

    int best_threads = baseline_threads;
    float best_ms = 1.0e30F;
    for (const int threads : candidates) {
        // Warm-up once, then use a median of three identical-state measurements.
        reset();
        (void)cn_time_mode<VariantIndex>(stream, mode, threads, sample_count, clone, clone_ctx);
        std::array<float, kCnBlockTunePasses> times{};
        for (int pass = 0; pass < kCnBlockTunePasses; ++pass) {
            reset();
            times[pass] = cn_time_mode<VariantIndex>(stream, mode, threads, sample_count, clone, clone_ctx);
        }
        const float median = cn_hardened_median(times);
        if (median < best_ms) {
            best_ms = median;
            best_threads = threads;
        }
    }

    cudaFree(clone_ctx);
    cudaFree(clone);
    g_cn_block_tuned_threads[device_id][VariantIndex] = best_threads;
    save_cn_block_tune_cache<VariantIndex>(device_id, props, count, mode, best_threads);

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN block tuner] GPU " << device_id
              << " | " << cfg.name
              << " | mode=" << cn_residency_mode_name(mode)
              << " | baseline=" << baseline_threads
              << " | selected=" << best_threads
              << " | median=" << best_ms << " ms"
              << " | candidates=" << candidates.size()
              << std::defaultfloat << '\n';
    return best_threads;
}

template <std::uint8_t VariantIndex, bool UseTTable>
void launch_cn_loop_block_tuned(cudaStream_t stream,
                                std::uint8_t* states,
                                std::size_t count,
                                std::uint8_t* scratchpads,
                                cryptonight::SplitContext* contexts,
                                const CnGeometry& geometry)
{
    (void)states;
    (void)geometry;
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN block tuner failed");
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) {
        cn_launch_mode<VariantIndex>(stream, 4, 128, count, scratchpads, contexts);
        return;
    }

    cn_hardened_autoselect_once<VariantIndex>(stream, device_id, count, scratchpads, contexts);
    int mode = g_cn_hardened_mode[device_id][VariantIndex];
    int baseline_threads = g_cn_hardened_threads[device_id][VariantIndex];
    if (mode == 0) mode = 4;
    if (baseline_threads <= 0) baseline_threads = 128;

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN block tuner failed");
    const int threads = cn_tune_selected_block_size<VariantIndex>(
        stream, device_id, props, mode, baseline_threads, count, scratchpads, contexts);
    cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
}

} // anonymous namespace
