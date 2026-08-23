// Per-device production selector for the cooperative CryptoNight loop.
//
// The cooperative backend is only eligible after full final-state parity passes
// and it beats the already-tuned production loop by at least 2%. The selected
// backend and geometry are cached per GPU / driver / runtime / batch.

namespace {

constexpr int kCnCoopCacheRevision = 1;
constexpr float kCnCoopRequiredSpeedup = 0.98F;
static std::array<std::array<bool, 6>, kMaxLoopTuneDevices> g_cn_coop_enabled{};
static std::array<std::array<int, 6>, kMaxLoopTuneDevices> g_cn_coop_threads{};
static std::array<bool, kMaxLoopTuneDevices> g_cn_coop_ready{};
static std::array<std::array<bool, 6>, kMaxLoopTuneDevices> g_cn_coop_runtime_reported{};

bool cn_coop_disabled_by_env()
{
    const char* value = std::getenv("YERBAS_CUDA_DISABLE_COOP");
    return value && *value && std::string(value) != "0";
}

std::filesystem::path cn_coop_cache_path(int device_id,
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
    return cn_cache_directory() / ("cn-coop-rev" + std::to_string(kCnCoopCacheRevision) + '-' + key + ".txt");
}

bool load_cn_coop_cache(int device_id,
                        const cudaDeviceProp& props,
                        std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_coop_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision) || magic != "YERBAS_CN_COOP" || revision != kCnCoopCacheRevision)
        return false;

    std::array<bool, 6> enabled{};
    std::array<int, 6> threads{};
    for (std::size_t i = 0; i < 6; ++i) {
        std::size_t variant = 0;
        int enabled_int = 0;
        int thread_count = 0;
        if (!(in >> variant >> enabled_int >> thread_count) || variant != i ||
            (enabled_int != 0 && enabled_int != 1) ||
            (thread_count != 32 && thread_count != 64 && thread_count != 128 && thread_count != 256)) {
            return false;
        }
        enabled[i] = enabled_int != 0;
        threads[i] = thread_count;
    }

    g_cn_coop_enabled[device_id] = enabled;
    g_cn_coop_threads[device_id] = threads;
    g_cn_coop_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CryptoNight cooperative cache loaded | " << path.string() << '\n';
    return true;
}

void save_cn_coop_cache(int device_id,
                        const cudaDeviceProp& props,
                        std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || !g_cn_coop_ready[device_id]) return;
    try {
        const auto path = cn_coop_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_COOP " << kCnCoopCacheRevision << '\n';
        for (std::size_t i = 0; i < 6; ++i)
            out << i << ' ' << (g_cn_coop_enabled[device_id][i] ? 1 : 0)
                << ' ' << g_cn_coop_threads[device_id][i] << '\n';
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
            std::cout << "[GPU " << device_id << "] CryptoNight cooperative cache saved | " << path.string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] CryptoNight cooperative cache write skipped: " << e.what() << '\n';
    }
}

template <std::uint8_t V>
bool validate_cn_coop_final_state(int device_id,
                                  cudaStream_t stream,
                                  std::uint8_t* states,
                                  std::uint8_t* scratch,
                                  cryptonight::SplitContext* ctx,
                                  const CnGeometry& g)
{
    constexpr std::size_t count = 2U;
    std::array<std::uint8_t, count * kStateBytes> reference{};
    std::array<std::uint8_t, count * kStateBytes> cooperative{};
    const int setup_blocks = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    const int final_blocks = static_cast<int>((count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                              static_cast<std::size_t>(g.final_threads));

    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x5bU + V * 17U), count * kStateBytes, stream),
                   "coop final-state parity reset failed");
        if (g.aes_backend == 1)
            cryptonight_setup_stage<V, true><<<setup_blocks, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_setup_stage<V, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop final-state parity setup failed");
    };
    auto finalize = [&]() {
        if (g.aes_backend == 1)
            cryptonight_final_stage<V, true><<<final_blocks, g.final_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_final_stage<V, false><<<final_blocks, g.final_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop final-state parity final failed");
    };

    prepare();
    launch_cn_selected_single<V>(stream, device_id, g.loop_threads, count, scratch, ctx);
    check_cuda(cudaGetLastError(), "coop final-state reference loop failed");
    finalize();
    check_cuda(cudaMemcpyAsync(reference.data(), states, reference.size(), cudaMemcpyDeviceToHost, stream),
               "coop final-state reference copy failed");
    check_cuda(cudaStreamSynchronize(stream), "coop final-state reference sync failed");

    prepare();
    constexpr int threads = 32;
    const int blocks = static_cast<int>((count * 4U + threads - 1U) / threads);
    cryptonight_loop_stage_coop4<V><<<blocks, threads, 0, stream>>>(count, scratch, ctx);
    check_cuda(cudaGetLastError(), "coop final-state candidate loop failed");
    finalize();
    check_cuda(cudaMemcpyAsync(cooperative.data(), states, cooperative.size(), cudaMemcpyDeviceToHost, stream),
               "coop final-state candidate copy failed");
    check_cuda(cudaStreamSynchronize(stream), "coop final-state candidate sync failed");

    return reference == cooperative;
}

template <std::uint8_t V>
void tune_cn_coop_variant(int device_id,
                          const cudaDeviceProp& props,
                          cudaStream_t stream,
                          std::uint8_t* states,
                          std::size_t n,
                          std::uint8_t* scratch,
                          cryptonight::SplitContext* ctx,
                          const CnGeometry& g)
{
    const bool parity_ok = validate_cn_coop_final_state<V>(device_id, stream, states, scratch, ctx, g);
    const int setup_blocks = static_cast<int>((n + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x63U + V * 13U), n * kStateBytes, stream),
                   "coop production tune reset failed");
        if (g.aes_backend == 1)
            cryptonight_setup_stage<V, true><<<setup_blocks, g.setup_threads, 0, stream>>>(states, n, scratch, ctx);
        else
            cryptonight_setup_stage<V, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, n, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop production tune setup failed");
    };

    prepare();
    const float production_ms = time_cuda_phase(stream, [&]() {
        launch_cn_selected_single<V>(stream, device_id, g.loop_threads, n, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop production baseline failed");
    });

    float best_ms = 1.0e30F;
    int best_threads = 32;
    if (parity_ok) {
        constexpr std::array<int, 4> candidates{{32, 64, 128, 256}};
        for (const int threads : candidates) {
            if (threads > props.maxThreadsPerBlock) continue;
            prepare();
            const int blocks = static_cast<int>((n * 4U + static_cast<std::size_t>(threads) - 1U) /
                                                static_cast<std::size_t>(threads));
            const float ms = time_cuda_phase(stream, [&]() {
                cryptonight_loop_stage_coop4<V><<<blocks, threads, 0, stream>>>(n, scratch, ctx);
                check_cuda(cudaGetLastError(), "coop production candidate failed");
            });
            std::cout << std::fixed << std::setprecision(3)
                      << "[CUDA CN coop test] " << cryptonight::config_value(V).name
                      << " | threads=" << threads << " | " << ms << " ms"
                      << std::defaultfloat << '\n';
            if (ms < best_ms) {
                best_ms = ms;
                best_threads = threads;
            }
        }
    }

    const bool use_coop = parity_ok && best_ms < production_ms * kCnCoopRequiredSpeedup;
    g_cn_coop_enabled[device_id][V] = use_coop;
    g_cn_coop_threads[device_id][V] = best_threads;
    const float gain = parity_ok ? (production_ms / best_ms - 1.0F) * 100.0F : 0.0F;

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN coop autotune] " << cryptonight::config_value(V).name
              << " | final_parity=" << (parity_ok ? "PASS" : "FAIL")
              << " | production=" << production_ms << " ms";
    if (parity_ok)
        std::cout << " | coop=" << best_ms << " ms"
                  << " | threads=" << best_threads
                  << " | gain=" << gain << "%";
    std::cout << " | selected=" << (use_coop ? "coop4" : "production")
              << std::defaultfloat << '\n';
}

void ensure_cn_coop_modes(int device_id,
                          const cudaDeviceProp& props,
                          cudaStream_t stream,
                          std::uint8_t* states,
                          std::size_t batch_size,
                          std::uint8_t* scratch,
                          cryptonight::SplitContext* ctx,
                          const std::array<CnGeometry, 6>& geometries)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return;
    if (cn_coop_disabled_by_env()) {
        g_cn_coop_enabled[device_id].fill(false);
        g_cn_coop_threads[device_id].fill(32);
        g_cn_coop_ready[device_id] = true;
        std::cout << "[GPU " << device_id << "] cooperative CryptoNight backend disabled by YERBAS_CUDA_DISABLE_COOP\n";
        return;
    }
    if (load_cn_coop_cache(device_id, props, batch_size)) return;

    std::cout << "[GPU " << device_id << "] cooperative CryptoNight production autotune starting"
              << " | hashes=" << batch_size << " | candidates=32,64,128,256 | required_gain=2%\n";
    tune_cn_coop_variant<0>(device_id, props, stream, states, batch_size, scratch, ctx, geometries[0]);
    tune_cn_coop_variant<1>(device_id, props, stream, states, batch_size, scratch, ctx, geometries[1]);
    tune_cn_coop_variant<2>(device_id, props, stream, states, batch_size, scratch, ctx, geometries[2]);
    tune_cn_coop_variant<3>(device_id, props, stream, states, batch_size, scratch, ctx, geometries[3]);
    tune_cn_coop_variant<4>(device_id, props, stream, states, batch_size, scratch, ctx, geometries[4]);
    tune_cn_coop_variant<5>(device_id, props, stream, states, batch_size, scratch, ctx, geometries[5]);
    g_cn_coop_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] cooperative CryptoNight production autotune complete\n";
    save_cn_coop_cache(device_id, props, batch_size);
}

} // anonymous namespace

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    auto result = autotune_cn_geometries_prod(device_id, stream, states, batch_size,
                                              scratchpads, contexts);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties cooperative production tuner failed");
    ensure_cn_coop_modes(device_id, props, stream, states, batch_size,
                         scratchpads, contexts, result);
    return result;
}
