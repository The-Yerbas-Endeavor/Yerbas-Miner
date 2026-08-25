// Generic cooperative CryptoNight implementation selector.
//
// This layer is intentionally hardware-agnostic: for every CryptoNight variant
// that already qualified for cooperative execution, compare the established
// coop4-v1 kernel against corrected register-distributed coop4-v3 at the
// already-selected production thread count. v3 is eligible only after exact
// final-state parity and a >=2% loop-time win. Decisions are cached by the
// existing GPU/driver/runtime/batch key; no GPU model is hard-coded.

namespace {

constexpr int kCnCoopImplCacheRevision = 2;
constexpr float kCnCoopImplRequiredSpeedup = 0.98F;
static std::array<std::array<int, 6>, kMaxLoopTuneDevices> g_cn_coop_impl{}; // 1=v1, 3=v3
static std::array<bool, kMaxLoopTuneDevices> g_cn_coop_impl_ready{};

std::filesystem::path cn_coop_impl_cache_path(int device_id,
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
    return cn_cache_directory() / ("cn-coop-impl-rev" + std::to_string(kCnCoopImplCacheRevision) + '-' + key + ".txt");
}

bool load_cn_coop_impl_cache(int device_id,
                             const cudaDeviceProp& props,
                             std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_coop_impl_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision) || magic != "YERBAS_CN_COOP_IMPL" || revision != kCnCoopImplCacheRevision)
        return false;

    std::array<int, 6> impl{};
    for (std::size_t i = 0; i < impl.size(); ++i) {
        std::size_t variant = 0;
        int value = 1;
        if (!(in >> variant >> value) || variant != i || (value != 1 && value != 3)) return false;
        impl[i] = value;
    }
    g_cn_coop_impl[device_id] = impl;
    g_cn_coop_impl_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] cooperative kernel implementation cache loaded | "
              << path.string() << '\n';
    return true;
}

void save_cn_coop_impl_cache(int device_id,
                             const cudaDeviceProp& props,
                             std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || !g_cn_coop_impl_ready[device_id]) return;
    try {
        const auto path = cn_coop_impl_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_COOP_IMPL " << kCnCoopImplCacheRevision << '\n';
        for (std::size_t i = 0; i < 6; ++i)
            out << i << ' ' << g_cn_coop_impl[device_id][i] << '\n';
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
        // Non-fatal: next launch simply benchmarks again.
    }
}

template <std::uint8_t V>
void launch_cn_coop_impl(cudaStream_t stream,
                         int impl,
                         int threads,
                         std::size_t count,
                         std::uint8_t* scratch,
                         cryptonight::SplitContext* ctx)
{
    const int blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    if (impl == 3)
        cryptonight_loop_stage_coop4_v3<V><<<blocks, threads, 0, stream>>>(count, scratch, ctx);
    else
        cryptonight_loop_stage_coop4<V><<<blocks, threads, 0, stream>>>(count, scratch, ctx);
}

template <std::uint8_t V>
bool validate_cn_coop_impl_v3(cudaStream_t stream,
                              std::uint8_t* states,
                              std::uint8_t* scratch,
                              cryptonight::SplitContext* ctx,
                              const CnGeometry& g,
                              int threads)
{
    constexpr std::size_t count = 2U;
    std::array<std::uint8_t, count * kStateBytes> v1{};
    std::array<std::uint8_t, count * kStateBytes> v3{};

    auto run = [&](int impl, std::array<std::uint8_t, count * kStateBytes>& output) {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x71U + V * 7U), count * kStateBytes, stream),
                   "coop impl parity reset failed");
        const int sb = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                        static_cast<std::size_t>(g.setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<V, true><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_setup_stage<V, false><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop impl parity setup failed");

        launch_cn_coop_impl<V>(stream, impl, threads, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop impl parity loop failed");

        const int fb = static_cast<int>((count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                        static_cast<std::size_t>(g.final_threads));
        if (g.aes_backend == 1)
            cryptonight_final_stage<V, true><<<fb, g.final_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_final_stage<V, false><<<fb, g.final_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop impl parity final failed");
        check_cuda(cudaMemcpyAsync(output.data(), states, output.size(), cudaMemcpyDeviceToHost, stream),
                   "coop impl parity copy failed");
        check_cuda(cudaStreamSynchronize(stream), "coop impl parity sync failed");
    };

    run(1, v1);
    run(3, v3);
    return v1 == v3;
}

template <std::uint8_t V>
void tune_cn_coop_impl_variant(int device_id,
                               cudaStream_t stream,
                               std::uint8_t* states,
                               std::size_t count,
                               std::uint8_t* scratch,
                               cryptonight::SplitContext* ctx,
                               const CnGeometry& g)
{
    g_cn_coop_impl[device_id][V] = 1;
    if (!g_cn_coop_enabled[device_id][V]) return;

    int threads = g_cn_coop_threads[device_id][V];
    if (threads <= 0) threads = 32;
    const bool parity = validate_cn_coop_impl_v3<V>(stream, states, scratch, ctx, g, threads);
    if (!parity) {
        std::cout << "[CUDA CN coop impl] " << cryptonight::config_value(V).name
                  << " | v3 parity=FAIL | selected=coop4-v1\n";
        return;
    }

    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x79U + V * 11U), count * kStateBytes, stream),
                   "coop impl benchmark reset failed");
        const int sb = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                        static_cast<std::size_t>(g.setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<V, true><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_setup_stage<V, false><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop impl benchmark setup failed");
    };

    prepare();
    const float v1_ms = time_cuda_phase(stream, [&]() {
        launch_cn_coop_impl<V>(stream, 1, threads, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop impl v1 benchmark failed");
    });
    prepare();
    const float v3_ms = time_cuda_phase(stream, [&]() {
        launch_cn_coop_impl<V>(stream, 3, threads, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop impl v3 benchmark failed");
    });

    const bool use_v3 = v3_ms < v1_ms * kCnCoopImplRequiredSpeedup;
    g_cn_coop_impl[device_id][V] = use_v3 ? 3 : 1;
    const float gain = (v1_ms / v3_ms - 1.0F) * 100.0F;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN coop impl] " << cryptonight::config_value(V).name
              << " | threads=" << threads
              << " | v1=" << v1_ms << " ms"
              << " | v3=" << v3_ms << " ms"
              << " | parity=PASS"
              << " | gain=" << gain << "%"
              << " | selected=" << (use_v3 ? "coop4-v3" : "coop4-v1")
              << std::defaultfloat << '\n';
}

void ensure_cn_coop_impl_modes(int device_id,
                               const cudaDeviceProp& props,
                               cudaStream_t stream,
                               std::uint8_t* states,
                               std::size_t batch_size,
                               std::uint8_t* scratch,
                               cryptonight::SplitContext* ctx,
                               const std::array<CnGeometry, 6>& geometries)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return;
    if (load_cn_coop_impl_cache(device_id, props, batch_size)) return;

    g_cn_coop_impl[device_id].fill(1);
    tune_cn_coop_impl_variant<0>(device_id, stream, states, batch_size, scratch, ctx, geometries[0]);
    tune_cn_coop_impl_variant<1>(device_id, stream, states, batch_size, scratch, ctx, geometries[1]);
    tune_cn_coop_impl_variant<2>(device_id, stream, states, batch_size, scratch, ctx, geometries[2]);
    tune_cn_coop_impl_variant<3>(device_id, stream, states, batch_size, scratch, ctx, geometries[3]);
    tune_cn_coop_impl_variant<4>(device_id, stream, states, batch_size, scratch, ctx, geometries[4]);
    tune_cn_coop_impl_variant<5>(device_id, stream, states, batch_size, scratch, ctx, geometries[5]);
    g_cn_coop_impl_ready[device_id] = true;
    save_cn_coop_impl_cache(device_id, props, batch_size);
}

} // anonymous namespace
