// Targeted generic CN-Fast v5 selector.
//
// v5 tests the profiler's dependent-load/multiply/writeback hotspot using one
// aligned uint4 load and one aligned uint4 writeback in lane 0. It is compared
// against the already-selected CN-Fast cooperative path (v4 if qualified,
// otherwise the generic coop implementation). Exact final-state parity and a
// >=2% loop-time improvement are required. No GPU model is hard-coded.

namespace {

constexpr int kCnFastV5CacheRevision = 1;
constexpr float kCnFastV5RequiredSpeedup = 0.98F;
static std::array<bool, kMaxLoopTuneDevices> g_cn_fast_v5_enabled{};
static std::array<bool, kMaxLoopTuneDevices> g_cn_fast_v5_ready{};

std::filesystem::path cn_fast_v5_cache_path(int device_id,
                                            const cudaDeviceProp& props,
                                            std::size_t batch_size)
{
    int driver_version = 0, runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, batch_size, driver_version, runtime_version);
    return cn_cache_directory() / ("cn-fast-v5-rev" + std::to_string(kCnFastV5CacheRevision) + '-' + key + ".txt");
}

bool load_cn_fast_v5_cache(int device_id,
                           const cudaDeviceProp& props,
                           std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    const char* benchmark = std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if ((retune && *retune && std::string(retune) != "0") ||
        (benchmark && *benchmark && std::string(benchmark) != "0")) return false;
    const auto path = cn_fast_v5_cache_path(device_id, props, batch_size);
    std::ifstream in(path);
    if (!in) return false;
    std::string magic;
    int revision = 0, enabled = 0;
    if (!(in >> magic >> revision >> enabled) || magic != "YERBAS_CN_FAST_V5" ||
        revision != kCnFastV5CacheRevision || (enabled != 0 && enabled != 1)) return false;
    g_cn_fast_v5_enabled[device_id] = enabled != 0;
    g_cn_fast_v5_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CN-Fast v5 cache | selected="
              << (enabled ? "coop4-v5" : "existing") << '\n';
    return true;
}

void save_cn_fast_v5_cache(int device_id,
                           const cudaDeviceProp& props,
                           std::size_t batch_size)
{
    try {
        const auto path = cn_fast_v5_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (out) out << "YERBAS_CN_FAST_V5 " << kCnFastV5CacheRevision << ' '
                     << (g_cn_fast_v5_enabled[device_id] ? 1 : 0) << '\n';
    } catch (...) {}
}

void launch_cn_fast_v5(cudaStream_t stream,
                       int threads,
                       std::size_t count,
                       std::uint8_t* scratch,
                       cryptonight::SplitContext* ctx)
{
    const int blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    cryptonight_loop_stage_coop4_v5<2><<<blocks, threads, 0, stream>>>(count, scratch, ctx);
}

void launch_cn_fast_existing_for_v5(int device_id,
                                    cudaStream_t stream,
                                    int threads,
                                    std::size_t count,
                                    std::uint8_t* scratch,
                                    cryptonight::SplitContext* ctx)
{
    if (device_id >= 0 && device_id < kMaxLoopTuneDevices &&
        g_cn_fast_v4_ready[device_id] && g_cn_fast_v4_enabled[device_id]) {
        launch_cn_fast_v4(stream, threads, count, scratch, ctx);
        return;
    }
    int impl = 1;
    if (device_id >= 0 && device_id < kMaxLoopTuneDevices && g_cn_coop_impl_ready[device_id])
        impl = g_cn_coop_impl[device_id][2];
    launch_cn_coop_impl<2>(stream, impl, threads, count, scratch, ctx);
}

bool validate_cn_fast_v5(int device_id,
                         cudaStream_t stream,
                         std::uint8_t* states,
                         std::uint8_t* scratch,
                         cryptonight::SplitContext* ctx,
                         const CnGeometry& g,
                         int threads)
{
    constexpr std::size_t count = 2U;
    std::array<std::uint8_t, count * kStateBytes> baseline{};
    std::array<std::uint8_t, count * kStateBytes> candidate{};
    auto run = [&](bool v5, std::array<std::uint8_t, count * kStateBytes>& out) {
        check_cuda(cudaMemsetAsync(states, 0x68, count * kStateBytes, stream), "CN-Fast v5 parity reset failed");
        const int sb = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                        static_cast<std::size_t>(g.setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<2, true><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_setup_stage<2, false><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "CN-Fast v5 parity setup failed");
        if (v5) launch_cn_fast_v5(stream, threads, count, scratch, ctx);
        else launch_cn_fast_existing_for_v5(device_id, stream, threads, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "CN-Fast v5 parity loop failed");
        const int fb = static_cast<int>((count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                        static_cast<std::size_t>(g.final_threads));
        if (g.aes_backend == 1)
            cryptonight_final_stage<2, true><<<fb, g.final_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_final_stage<2, false><<<fb, g.final_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "CN-Fast v5 parity final failed");
        check_cuda(cudaMemcpyAsync(out.data(), states, out.size(), cudaMemcpyDeviceToHost, stream),
                   "CN-Fast v5 parity copy failed");
        check_cuda(cudaStreamSynchronize(stream), "CN-Fast v5 parity sync failed");
    };
    run(false, baseline);
    run(true, candidate);
    return baseline == candidate;
}

void ensure_cn_fast_v5_mode(int device_id,
                            const cudaDeviceProp& props,
                            cudaStream_t stream,
                            std::uint8_t* states,
                            std::size_t count,
                            std::uint8_t* scratch,
                            cryptonight::SplitContext* ctx,
                            const CnGeometry& g)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || !g_cn_coop_enabled[device_id][2]) return;
    if (load_cn_fast_v5_cache(device_id, props, count)) return;
    g_cn_fast_v5_enabled[device_id] = false;
    g_cn_fast_v5_ready[device_id] = true;

    int threads = g_cn_coop_threads[device_id][2];
    if (threads <= 0) threads = 32;
    const bool parity = validate_cn_fast_v5(device_id, stream, states, scratch, ctx, g, threads);
    if (!parity) {
        std::cout << "[CUDA CN-Fast v5] parity=FAIL | selected=existing\n";
        save_cn_fast_v5_cache(device_id, props, count);
        return;
    }

    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, 0x83, count * kStateBytes, stream), "CN-Fast v5 benchmark reset failed");
        const int sb = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                        static_cast<std::size_t>(g.setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<2, true><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_setup_stage<2, false><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "CN-Fast v5 benchmark setup failed");
    };

    prepare();
    const float baseline_ms = time_cuda_phase(stream, [&]() {
        launch_cn_fast_existing_for_v5(device_id, stream, threads, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "CN-Fast existing-v5 benchmark failed");
    });
    prepare();
    const float v5_ms = time_cuda_phase(stream, [&]() {
        launch_cn_fast_v5(stream, threads, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "CN-Fast v5 benchmark failed");
    });

    const float gain = (baseline_ms / v5_ms - 1.0F) * 100.0F;
    const bool use_v5 = v5_ms < baseline_ms * kCnFastV5RequiredSpeedup;
    g_cn_fast_v5_enabled[device_id] = use_v5;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN-Fast v5] GPU " << device_id
              << " | threads=" << threads
              << " | baseline=" << baseline_ms << " ms"
              << " | v5=" << v5_ms << " ms"
              << " | parity=PASS | gain=" << gain << "%"
              << " | selected=" << (use_v5 ? "coop4-v5" : "existing")
              << std::defaultfloat << '\n';
    save_cn_fast_v5_cache(device_id, props, count);
}

} // anonymous namespace
