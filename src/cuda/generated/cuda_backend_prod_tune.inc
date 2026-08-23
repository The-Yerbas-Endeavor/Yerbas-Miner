// Production-batch CryptoNight loop geometry tuner.
//
// The legacy tuner intentionally uses a small sample to keep startup bounded.
// That is useful for setup/final geometry, but the data-dependent CN memory loop
// can choose a different optimum once the GPU is saturated by the real batch.
// This second-stage tuner measures the thread-count/unroll pair together at the
// configured production batch size and persists the winner per GPU/CN variant.

constexpr int kCnProdLoopCacheRevision = 1;
static std::array<std::array<int, 6>, kMaxLoopTuneDevices> g_cn_prod_threads{};
static std::array<std::array<int, 6>, kMaxLoopTuneDevices> g_cn_prod_unroll{};
static std::array<bool, kMaxLoopTuneDevices> g_cn_prod_ready{};

std::filesystem::path cn_prod_loop_cache_path(int device_id,
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
    return cn_cache_directory() / ("cn-prod-loop-rev" + std::to_string(kCnProdLoopCacheRevision) + '-' + key + ".txt");
}

bool load_cn_prod_loop_cache(int device_id,
                             const cudaDeviceProp& props,
                             std::size_t batch_size,
                             std::array<CnGeometry, 6>& geometries)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_prod_loop_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision) || magic != "YERBAS_CN_PROD_LOOP" || revision != kCnProdLoopCacheRevision)
        return false;

    std::array<int, 6> threads{};
    std::array<int, 6> unroll{};
    for (std::size_t i = 0; i < 6; ++i) {
        std::size_t variant = 0;
        int t = 0;
        int u = 0;
        if (!(in >> variant >> t >> u) || variant != i ||
            (t != 32 && t != 64 && t != 128 && t != 256) ||
            (u != 1 && u != 2 && u != 4)) {
            return false;
        }
        threads[i] = t;
        unroll[i] = u;
    }

    g_cn_prod_threads[device_id] = threads;
    g_cn_prod_unroll[device_id] = unroll;
    g_cn_prod_ready[device_id] = true;
    for (std::size_t i = 0; i < 6; ++i) {
        geometries[i].loop_threads = threads[i];
        g_cn_loop_unroll[device_id][i] = unroll[i];
    }
    g_cn_loop_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CryptoNight production loop cache loaded | " << path.string() << '\n';
    return true;
}

void save_cn_prod_loop_cache(int device_id,
                             const cudaDeviceProp& props,
                             std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || !g_cn_prod_ready[device_id]) return;
    try {
        const auto path = cn_prod_loop_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_PROD_LOOP " << kCnProdLoopCacheRevision << '\n';
        for (std::size_t i = 0; i < 6; ++i)
            out << i << ' ' << g_cn_prod_threads[device_id][i] << ' '
                << g_cn_prod_unroll[device_id][i] << '\n';
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
            std::cout << "[GPU " << device_id << "] CryptoNight production loop cache saved | " << path.string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] CryptoNight production loop cache write skipped: " << e.what() << '\n';
    }
}

template <std::uint8_t VariantIndex>
void tune_cn_prod_variant(int device_id,
                          const cudaDeviceProp& props,
                          cudaStream_t stream,
                          std::uint8_t* states,
                          std::size_t batch_size,
                          std::uint8_t* scratchpads,
                          cryptonight::SplitContext* contexts,
                          CnGeometry& g)
{
    constexpr std::array<int, 4> thread_candidates{{32, 64, 128, 256}};
    constexpr std::array<int, 3> unroll_candidates{{1, 2, 4}};
    const bool use_ttable = g_cn_loop_aes_ready[device_id] && g_cn_loop_ttable[device_id][VariantIndex];

    float best_ms = 1.0e30F;
    int best_threads = g.loop_threads;
    int best_unroll = g_cn_loop_ready[device_id] ? g_cn_loop_unroll[device_id][VariantIndex] : 1;

    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + VariantIndex * 17U),
                                   batch_size * kStateBytes, stream),
                   "cudaMemsetAsync production CN tune states failed");
        const int setup_threads = g.setup_threads;
        const int setup_blocks = static_cast<int>((batch_size + static_cast<std::size_t>(setup_threads) - 1U) /
                                                  static_cast<std::size_t>(setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, setup_threads, 0, stream>>>(
                states, batch_size, scratchpads, contexts);
        else
            cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(
                states, batch_size, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "production CN tune setup failed");
    };

    for (const int threads : thread_candidates) {
        if (threads > props.maxThreadsPerBlock) continue;
        const int blocks = static_cast<int>((batch_size + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        for (const int unroll : unroll_candidates) {
            prepare();
            const float ms = time_cuda_phase(stream, [&]() {
                launch_cn_loop_backend<VariantIndex>(stream, unroll, use_ttable, blocks, threads,
                                                     batch_size, scratchpads, contexts);
                check_cuda(cudaGetLastError(), "production CN tune loop launch failed");
            });
            std::cout << std::fixed << std::setprecision(3)
                      << "[CUDA CN production test] " << cryptonight::config_value(VariantIndex).name
                      << " | hashes=" << batch_size
                      << " | threads=" << threads
                      << " | unroll=" << unroll
                      << " | loop_aes=" << (use_ttable ? "ttable" : "portable")
                      << " | " << ms << " ms"
                      << std::defaultfloat << '\n';
            if (ms < best_ms) {
                best_ms = ms;
                best_threads = threads;
                best_unroll = unroll;
            }
        }
    }

    g.loop_threads = best_threads;
    g_cn_prod_threads[device_id][VariantIndex] = best_threads;
    g_cn_prod_unroll[device_id][VariantIndex] = best_unroll;
    g_cn_loop_unroll[device_id][VariantIndex] = best_unroll;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN production tune] " << cryptonight::config_value(VariantIndex).name
              << " | hashes=" << batch_size
              << " | selected=" << best_threads << 'x' << best_unroll
              << " | " << best_ms << " ms"
              << std::defaultfloat << '\n';
}

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    // Run the established small-sample tuner first. This preserves the proven
    // setup/final AES selections, loop AES production selection, and safety
    // validation. Then override only the loop threads/unroll pair using the
    // actual production batch.
    auto result = autotune_cn_geometries_legacy(device_id, stream, states, batch_size,
                                                scratchpads, contexts);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties production CryptoNight tuner failed");

    if (load_cn_prod_loop_cache(device_id, props, batch_size, result)) return result;

    std::cout << "[GPU " << device_id << "] CryptoNight production loop geometry autotune starting"
              << " | hashes=" << batch_size
              << " | threads=32,64,128,256 | unroll=1,2,4\n";
    tune_cn_prod_variant<0>(device_id, props, stream, states, batch_size, scratchpads, contexts, result[0]);
    tune_cn_prod_variant<1>(device_id, props, stream, states, batch_size, scratchpads, contexts, result[1]);
    tune_cn_prod_variant<2>(device_id, props, stream, states, batch_size, scratchpads, contexts, result[2]);
    tune_cn_prod_variant<3>(device_id, props, stream, states, batch_size, scratchpads, contexts, result[3]);
    tune_cn_prod_variant<4>(device_id, props, stream, states, batch_size, scratchpads, contexts, result[4]);
    tune_cn_prod_variant<5>(device_id, props, stream, states, batch_size, scratchpads, contexts, result[5]);
    g_cn_prod_ready[device_id] = true;
    g_cn_loop_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CryptoNight production loop geometry autotune complete\n";
    save_cn_prod_loop_cache(device_id, props, batch_size);
    return result;
}
