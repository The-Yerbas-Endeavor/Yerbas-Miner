// Runtime block-size tuner for the selected production CryptoNight phase-2 kernel.
//
// Kernel selection and exact parity remain owned by the hardened selector. This
// layer tunes launch geometry for the approved kernel and now learns the kernel
// mode itself from real production-sized launches. CUDA's occupancy recommendation
// is only the starting point; no GPU model or architecture table forces a choice.

namespace {

constexpr int kCnBlockTuneRevision = 1;
constexpr int kCnBlockTunePasses = 3;
constexpr int kCnProductionSelectorRevision = 1;
constexpr int kCnProductionSelectorPasses = 3;
constexpr float kCnProductionPromotionRatio = 0.95F;
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_block_tuned_threads{};

struct CnProductionSelectorState {
    bool initialized{false};
    bool selected{false};
    bool tile_ok{false};
    bool cg_ok{false};
    std::size_t count{0};
    int threads_ref{128};
    int threads_t4{128};
    int threads_tile{128};
    int threads_cg{128};
    int baseline_samples{0};
    int tile_samples{0};
    int cg_samples{0};
    std::array<float, kCnProductionSelectorPasses> baseline_ms{};
    std::array<float, kCnProductionSelectorPasses> tile_ms{};
    std::array<float, kCnProductionSelectorPasses> cg_ms{};
};

static std::array<std::array<CnProductionSelectorState, 6>, kCn2LaneMaxDevices>
    g_cn_production_selector{};

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
std::filesystem::path cn_production_selector_cache_path(int device_id,
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
        ("cn-production-rev" + std::to_string(kCnProductionSelectorRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_production_selector_cache(int device_id,
                                       const cudaDeviceProp& props,
                                       std::size_t count)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_production_selector_cache_path<VariantIndex>(device_id, props, count);
    if (path.empty()) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0, variant = -1, mode = 0, threads = 0;
    const int warp = std::max(32, props.warpSize);
    if (!(in >> magic >> revision >> variant >> mode >> threads) ||
        magic != "YERBAS_CN_PRODUCTION" || revision != kCnProductionSelectorRevision ||
        variant != static_cast<int>(VariantIndex) ||
        (mode != 4 && mode != 44 && mode != 222 && mode != 444) ||
        threads < warp || threads > props.maxThreadsPerBlock || (threads % warp) != 0)
        return false;

    if (mode == 222) {
        int active = 0;
        const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, cryptonight_loop_stage_ttable2_tile64<VariantIndex>, threads,
            cn_tile64_dynamic_shared_bytes(threads));
        if (rc != cudaSuccess || active <= 0) { cudaGetLastError(); return false; }
    }

    auto& state = g_cn_production_selector[device_id][VariantIndex];
    state.initialized = true;
    state.selected = true;
    state.count = count;
    g_cn_hardened_mode[device_id][VariantIndex] = mode;
    g_cn_hardened_threads[device_id][VariantIndex] = threads;
    std::cout << "[GPU " << device_id << "] CryptoNight production selector cache loaded | "
              << cryptonight::config_value(VariantIndex).name
              << " | batch=" << count
              << " | production=" << cn_residency_mode_name(mode)
              << " | threads=" << threads << '\n';
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_production_selector_cache(int device_id,
                                       const cudaDeviceProp& props,
                                       std::size_t count,
                                       int mode,
                                       int threads)
{
    try {
        const auto path = cn_production_selector_cache_path<VariantIndex>(device_id, props, count);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_PRODUCTION " << kCnProductionSelectorRevision << ' '
            << static_cast<int>(VariantIndex) << ' ' << mode << ' ' << threads << '\n';
        out.close();
        if (!out) return;
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (...) {}
}

template <std::uint8_t VariantIndex>
void initialize_cn_production_selector(cudaStream_t stream,
                                       int device_id,
                                       const cudaDeviceProp& props,
                                       std::size_t count,
                                       const std::uint8_t* scratchpads,
                                       const cryptonight::SplitContext* contexts)
{
    auto& state = g_cn_production_selector[device_id][VariantIndex];
    if (state.initialized) return;
    if (load_cn_production_selector_cache<VariantIndex>(device_id, props, count)) return;

    state = CnProductionSelectorState{};
    state.count = count;
    const int warp = std::max(32, props.warpSize);
    state.threads_ref = cn_occupancy_block_size(
        cryptonight_loop_stage_cooperative<VariantIndex>, warp, props.maxThreadsPerBlock);
    state.threads_t4 = cn_occupancy_block_size(
        cryptonight_loop_stage_ttable4_coalesced<VariantIndex>, warp, props.maxThreadsPerBlock);
    state.threads_cg = cn_occupancy_block_size(
        cryptonight_loop_stage_ttable4_cg<VariantIndex>, warp, props.maxThreadsPerBlock);
    state.threads_tile = cn_tile64_block_size<VariantIndex>(props);

    const bool parity_t4 = cn_validate_pair<VariantIndex>(
        stream, 4, 32, 44, 32, scratchpads, contexts);
    if (!parity_t4) {
        state.initialized = true;
        state.selected = true;
        g_cn_hardened_mode[device_id][VariantIndex] = 4;
        g_cn_hardened_threads[device_id][VariantIndex] = state.threads_ref;
        save_cn_production_selector_cache<VariantIndex>(
            device_id, props, count, 4, state.threads_ref);
        std::cout << "[CUDA CN production selector] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | batch=" << count
                  << " | baseline-parity=FAIL | production=4-lane-ref\n";
        return;
    }

    state.tile_ok = cn_validate_pair<VariantIndex>(
        stream, 44, 32, 222, 32, scratchpads, contexts);
    state.cg_ok = cn_validate_pair<VariantIndex>(
        stream, 44, 32, 444, 32, scratchpads, contexts);
    state.initialized = true;

    std::cout << "[CUDA CN production selector] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | batch=" << count
              << " | passes=" << kCnProductionSelectorPasses
              << " | threshold=5%"
              << " | tile64-parity=" << (state.tile_ok ? "PASS" : "FAIL")
              << " | cg-parity=" << (state.cg_ok ? "PASS" : "FAIL")
              << " | sampling=real-production-batch\n";
}

inline int cn_production_trial_mode(const CnProductionSelectorState& state)
{
    if (state.baseline_samples < kCnProductionSelectorPasses) return 44;
    if (state.tile_ok && state.tile_samples < kCnProductionSelectorPasses) return 222;
    if (state.cg_ok && state.cg_samples < kCnProductionSelectorPasses) return 444;
    return 44;
}

inline int cn_production_trial_threads(const CnProductionSelectorState& state, int mode)
{
    if (mode == 222) return state.threads_tile;
    if (mode == 444) return state.threads_cg;
    if (mode == 44) return state.threads_t4;
    return state.threads_ref;
}

template <std::uint8_t VariantIndex>
void record_cn_production_sample(int device_id,
                                 const cudaDeviceProp& props,
                                 int mode,
                                 float elapsed_ms)
{
    auto& state = g_cn_production_selector[device_id][VariantIndex];
    if (mode == 44 && state.baseline_samples < kCnProductionSelectorPasses)
        state.baseline_ms[state.baseline_samples++] = elapsed_ms;
    else if (mode == 222 && state.tile_samples < kCnProductionSelectorPasses)
        state.tile_ms[state.tile_samples++] = elapsed_ms;
    else if (mode == 444 && state.cg_samples < kCnProductionSelectorPasses)
        state.cg_ms[state.cg_samples++] = elapsed_ms;

    const bool done = state.baseline_samples >= kCnProductionSelectorPasses &&
        (!state.tile_ok || state.tile_samples >= kCnProductionSelectorPasses) &&
        (!state.cg_ok || state.cg_samples >= kCnProductionSelectorPasses);
    if (!done) return;

    const float baseline = cn_hardened_median(state.baseline_ms);
    const float tile = state.tile_ok ? cn_hardened_median(state.tile_ms) : -1.0F;
    const float cg = state.cg_ok ? cn_hardened_median(state.cg_ms) : -1.0F;
    int best_mode = 44;
    int best_threads = state.threads_t4;
    float best_ms = baseline;
    if (state.tile_ok && tile < baseline * kCnProductionPromotionRatio && tile < best_ms) {
        best_mode = 222;
        best_threads = state.threads_tile;
        best_ms = tile;
    }
    if (state.cg_ok && cg < baseline * kCnProductionPromotionRatio && cg < best_ms) {
        best_mode = 444;
        best_threads = state.threads_cg;
        best_ms = cg;
    }

    state.selected = true;
    g_cn_hardened_mode[device_id][VariantIndex] = best_mode;
    g_cn_hardened_threads[device_id][VariantIndex] = best_threads;
    save_cn_production_selector_cache<VariantIndex>(
        device_id, props, state.count, best_mode, best_threads);

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN production selector] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | batch=" << state.count
              << " | baseline=" << baseline << " ms@" << state.threads_t4
              << " | tile64=" << tile << " ms@" << state.threads_tile
              << " | cg=" << cg << " ms@" << state.threads_cg
              << " | production=" << cn_residency_mode_name(best_mode)
              << " | selected=" << best_ms << " ms"
              << std::defaultfloat << '\n';
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

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN block tuner failed");
    initialize_cn_production_selector<VariantIndex>(
        stream, device_id, props, count, scratchpads, contexts);
    auto& selector = g_cn_production_selector[device_id][VariantIndex];

    // Staggered execution can call this wrapper with two different partition
    // sizes for the same variant. Learn from the first real production count and
    // keep the other partition on the parity-proven baseline until selection is
    // complete; never reset the trial state just because the partner partition
    // has a different count.
    if (!selector.selected && selector.count != count) {
        cn_launch_mode<VariantIndex>(stream, 44, selector.threads_t4,
                                     count, scratchpads, contexts);
        return;
    }

    if (!selector.selected) {
        const int mode = cn_production_trial_mode(selector);
        const int threads = cn_production_trial_threads(selector, mode);
        cudaEvent_t start{}, stop{};
        check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault),
                   "cudaEventCreate production selector start failed");
        check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault),
                   "cudaEventCreate production selector stop failed");
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord production selector start failed");
        cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord production selector stop failed");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize production selector failed");
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "cudaEventElapsedTime production selector failed");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        record_cn_production_sample<VariantIndex>(device_id, props, mode, elapsed_ms);
        return;
    }

    int mode = g_cn_hardened_mode[device_id][VariantIndex];
    int baseline_threads = g_cn_hardened_threads[device_id][VariantIndex];
    if (mode == 0) mode = 4;
    if (baseline_threads <= 0) baseline_threads = 128;
    const int threads = cn_tune_selected_block_size<VariantIndex>(
        stream, device_id, props, mode, baseline_threads, count, scratchpads, contexts);
    cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
}

} // anonymous namespace
