    for (const int threads : candidates) {
        if (threads > max_threads) continue;
        reset_states();
        const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(best.setup_threads) - 1U) /
                                                  static_cast<std::size_t>(best.setup_threads));
        const int loop_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(best.loop_threads) - 1U) /
                                                 static_cast<std::size_t>(best.loop_threads));
        cryptonight_setup_stage<VariantIndex><<<setup_blocks, best.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CryptoNight autotune final setup prep failed");
        cryptonight_loop_stage<VariantIndex><<<loop_blocks, best.loop_threads, 0, stream>>>(sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CryptoNight autotune final loop prep failed");
        const int blocks = static_cast<int>((sample_count + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        const float ms = time_cuda_phase(stream, [&]() {
            cryptonight_final_stage<VariantIndex><<<blocks, threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
            check_cuda(cudaGetLastError(), "CryptoNight autotune final launch failed");
        });
        if (ms < best_final_ms) { best_final_ms = ms; best.final_threads = threads; }
    }
    best.setup_ms = best_setup_ms;
    best.loop_ms = best_loop_ms;
    best.final_ms = best_final_ms;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA autotune] " << cfg.name
              << " | sample=" << sample_count
              << " | setup=" << best.setup_threads << " (" << best.setup_ms << " ms)"
              << " | loop=" << best.loop_threads << " (" << best.loop_ms << " ms)"
              << " | final=" << best.final_threads << " (" << best.final_ms << " ms)"
              << std::defaultfloat << '\n';
    return best;
}

constexpr int kCnDualCacheRevision = 1;
static std::array<std::array<bool, 6>, kMaxLoopTuneDevices> g_cn_dual_enabled{};
static std::array<std::array<int, 6>, kMaxLoopTuneDevices> g_cn_dual_threads{};
static std::array<bool, kMaxLoopTuneDevices> g_cn_dual_ready{};

template <std::uint8_t VariantIndex, int Unroll>
__global__ void cryptonight_loop_stage_dual(std::size_t count,
                                            std::uint8_t* scratchpads,
                                            const cryptonight::SplitContext* contexts)
{
    const std::size_t pair_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t index0 = pair_index * 2U;
    if (index0 >= count) return;

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    const std::size_t stride = static_cast<std::size_t>(cfg.page_size);
    std::uint8_t* scratchpad0 = scratchpads + index0 * stride;
    const std::size_t index1 = index0 + 1U;

    if (index1 < count) {
        std::uint8_t* scratchpad1 = scratchpads + index1 * stride;
        cryptonight::split_memory_loop_dual_tuned<VariantIndex, Unroll>(
            scratchpad0, contexts[index0], scratchpad1, contexts[index1]);
    } else {
        cryptonight::split_memory_loop_tuned<VariantIndex, Unroll>(scratchpad0, contexts[index0]);
    }
}

template <std::uint8_t VariantIndex>
void launch_cn_dual_mode(cudaStream_t stream,
                         int unroll,
                         int threads,
                         std::size_t count,
                         std::uint8_t* scratchpads,
                         const cryptonight::SplitContext* contexts)
{
    const std::size_t pairs = (count + 1U) / 2U;
    const int blocks = static_cast<int>((pairs + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    if (unroll == 4)
        cryptonight_loop_stage_dual<VariantIndex, 4><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    else if (unroll == 2)
        cryptonight_loop_stage_dual<VariantIndex, 2><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    else
        cryptonight_loop_stage_dual<VariantIndex, 1><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
}

std::filesystem::path cn_dual_cache_path(int device_id,
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
    return cn_cache_directory() / ("cn-dual-rev" + std::to_string(kCnDualCacheRevision) + '-' + key + ".txt");
}

bool load_cn_dual_cache(int device_id,
                        const cudaDeviceProp& props,
                        std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_dual_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision) || magic != "YERBAS_CN_DUAL" || revision != kCnDualCacheRevision)
        return false;

    std::array<bool, 6> enabled{};
    std::array<int, 6> threads{};
    for (std::size_t i = 0; i < 6; ++i) {
        std::size_t variant = 0;
        int enabled_int = 0;
        int thread_count = 0;
        if (!(in >> variant >> enabled_int >> thread_count) || variant != i ||
            (enabled_int != 0 && enabled_int != 1) ||
            (thread_count != 32 && thread_count != 64)) {
            return false;
        }
        enabled[i] = enabled_int != 0;
        threads[i] = thread_count;
    }

    g_cn_dual_enabled[device_id] = enabled;
    g_cn_dual_threads[device_id] = threads;
    g_cn_dual_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CryptoNight dual-state cache loaded | " << path.string() << '\n';
    return true;
}

void save_cn_dual_cache(int device_id,
                        const cudaDeviceProp& props,
                        std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || !g_cn_dual_ready[device_id]) return;
    try {
        const auto path = cn_dual_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_DUAL " << kCnDualCacheRevision << '\n';
        for (std::size_t i = 0; i < 6; ++i)
            out << i << ' ' << (g_cn_dual_enabled[device_id][i] ? 1 : 0)
                << ' ' << g_cn_dual_threads[device_id][i] << '\n';
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
            std::cout << "[GPU " << device_id << "] CryptoNight dual-state cache saved | " << path.string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] CryptoNight dual-state cache write skipped: " << e.what() << '\n';
    }
}

template <std::uint8_t VariantIndex>
void launch_cn_selected_single(cudaStream_t stream,
                               int device_id,
                               int threads,
                               std::size_t count,
                               std::uint8_t* scratchpads,
                               const cryptonight::SplitContext* contexts)
{
    int unroll = 1;
    if (device_id >= 0 && device_id < kMaxLoopTuneDevices && g_cn_loop_ready[device_id])
        unroll = g_cn_loop_unroll[device_id][VariantIndex];
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    launch_cn_loop_mode<VariantIndex>(stream, unroll, blocks, threads, count, scratchpads, contexts);
}

template <std::uint8_t VariantIndex>
bool validate_cn_dual_variant(int device_id,
                              cudaStream_t stream,
                              std::uint8_t* states,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              const CnGeometry& g)
{
    constexpr std::size_t validation_count = 2U;
    std::array<std::uint8_t, validation_count * kStateBytes> single_result{};
    std::array<std::uint8_t, validation_count * kStateBytes> dual_result{};
    const int setup_blocks = static_cast<int>((validation_count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    const int final_blocks = static_cast<int>((validation_count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                              static_cast<std::size_t>(g.final_threads));
    const int unroll = (device_id >= 0 && device_id < kMaxLoopTuneDevices && g_cn_loop_ready[device_id])
        ? g_cn_loop_unroll[device_id][VariantIndex] : 1;

    auto reset_and_setup = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x71U + VariantIndex * 13U),
                                   validation_count * kStateBytes, stream),
                   "cudaMemsetAsync dual-state parity states failed");
        if (g.aes_backend == 1)
            cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, g.setup_threads, 0, stream>>>(
                states, validation_count, scratchpads, contexts);
        else
            cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(
                states, validation_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "dual-state parity setup launch failed");
    };
    auto finalize = [&]() {
        if (g.aes_backend == 1)
            cryptonight_final_stage<VariantIndex, true><<<final_blocks, g.final_threads, 0, stream>>>(
                states, validation_count, scratchpads, contexts);
        else
            cryptonight_final_stage<VariantIndex, false><<<final_blocks, g.final_threads, 0, stream>>>(
                states, validation_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "dual-state parity final launch failed");
    };

    reset_and_setup();
    launch_cn_selected_single<VariantIndex>(stream, device_id, g.loop_threads, validation_count,
                                            scratchpads, contexts);
    check_cuda(cudaGetLastError(), "dual-state parity single loop launch failed");
    finalize();
    check_cuda(cudaMemcpyAsync(single_result.data(), states, single_result.size(), cudaMemcpyDeviceToHost, stream),
               "dual-state parity single copy failed");
    check_cuda(cudaStreamSynchronize(stream), "dual-state parity single synchronize failed");

    reset_and_setup();
    launch_cn_dual_mode<VariantIndex>(stream, unroll, 32, validation_count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "dual-state parity dual loop launch failed");
    finalize();
    check_cuda(cudaMemcpyAsync(dual_result.data(), states, dual_result.size(), cudaMemcpyDeviceToHost, stream),
               "dual-state parity dual copy failed");
    check_cuda(cudaStreamSynchronize(stream), "dual-state parity dual synchronize failed");

    return single_result == dual_result;
}

template <std::uint8_t VariantIndex>
void autotune_cn_dual_variant(int device_id,
                              const cudaDeviceProp& props,
                              cudaStream_t stream,
                              std::uint8_t* states,
                              std::size_t sample_count,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              const CnGeometry& g)
{
    const bool parity_ok = validate_cn_dual_variant<VariantIndex>(device_id, stream, states,
                                                                  scratchpads, contexts, g);
    const int unroll = (device_id >= 0 && device_id < kMaxLoopTuneDevices && g_cn_loop_ready[device_id])
        ? g_cn_loop_unroll[device_id][VariantIndex] : 1;
    const int single_threads = g.loop_threads;

    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes, stream),
                   "cudaMemsetAsync dual-state autotune states failed");
        const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                                  static_cast<std::size_t>(g.setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, g.setup_threads, 0, stream>>>(
                states, sample_count, scratchpads, contexts);
        else
            cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(
                states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "dual-state autotune setup prep failed");
    };

    prepare();
    const float single_ms = time_cuda_phase(stream, [&]() {
        launch_cn_selected_single<VariantIndex>(stream, device_id, single_threads, sample_count,
                                                scratchpads, contexts);
        check_cuda(cudaGetLastError(), "dual-state autotune single launch failed");
    });

    float best_dual_ms = 1.0e30F;
    int best_dual_threads = 32;
    if (parity_ok) {
        constexpr std::array<int, 2> thread_candidates{{32, 64}};
        for (const int threads : thread_candidates) {
            if (threads > props.maxThreadsPerBlock) continue;
            prepare();
            const float ms = time_cuda_phase(stream, [&]() {
                launch_cn_dual_mode<VariantIndex>(stream, unroll, threads, sample_count,
                                                  scratchpads, contexts);
                check_cuda(cudaGetLastError(), "dual-state autotune dual launch failed");
            });
            std::cout << std::fixed << std::setprecision(3)
                      << "[CUDA dual-state test] " << cryptonight::config_value(VariantIndex).name
                      << " | threads=" << threads << " | " << ms << " ms"
                      << std::defaultfloat << '\n';
            if (ms < best_dual_ms) {
                best_dual_ms = ms;
                best_dual_threads = threads;
            }
        }
    }

    // Require a real margin so measurement noise cannot enable a higher-register
    // path that only appeared fractionally faster during startup.
    const bool use_dual = parity_ok && best_dual_ms < single_ms * 0.98F;
    g_cn_dual_enabled[device_id][VariantIndex] = use_dual;
    g_cn_dual_threads[device_id][VariantIndex] = best_dual_threads;

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA dual-state autotune] " << cryptonight::config_value(VariantIndex).name
              << " | parity=" << (parity_ok ? "PASS" : "FAIL")
              << " | single=" << single_ms << " ms";
    if (parity_ok)
        std::cout << " | dual=" << best_dual_ms << " ms"
                  << " | dual_threads=" << best_dual_threads;
    std::cout << " | selected=" << (use_dual ? "dual" : "single")
              << std::defaultfloat << '\n';
}

void ensure_cn_dual_modes(int device_id,
                          const cudaDeviceProp& props,
                          cudaStream_t stream,
                          std::uint8_t* states,
                          std::size_t sample_count,
                          std::size_t batch_size,
                          std::uint8_t* scratchpads,
                          cryptonight::SplitContext* contexts,
                          const std::array<CnGeometry, 6>& geometries)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return;
    if (load_cn_dual_cache(device_id, props, batch_size)) return;

    std::cout << "[GPU " << device_id << "] empirical CryptoNight dual-state latency-hiding autotune starting\n";
    autotune_cn_dual_variant<0>(device_id, props, stream, states, sample_count, scratchpads, contexts, geometries[0]);
    autotune_cn_dual_variant<1>(device_id, props, stream, states, sample_count, scratchpads, contexts, geometries[1]);
    autotune_cn_dual_variant<2>(device_id, props, stream, states, sample_count, scratchpads, contexts, geometries[2]);
    autotune_cn_dual_variant<3>(device_id, props, stream, states, sample_count, scratchpads, contexts, geometries[3]);
    autotune_cn_dual_variant<4>(device_id, props, stream, states, sample_count, scratchpads, contexts, geometries[4]);
    autotune_cn_dual_variant<5>(device_id, props, stream, states, sample_count, scratchpads, contexts, geometries[5]);
    g_cn_dual_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] empirical CryptoNight dual-state latency-hiding autotune complete\n";
    save_cn_dual_cache(device_id, props, batch_size);
}

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    std::array<CnGeometry, 6> result{};
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CryptoNight autotune failed");
    std::size_t sample_count = std::max<std::size_t>(32U,
        static_cast<std::size_t>(std::max(1, props.multiProcessorCount)) * 2U);
    sample_count = std::min(sample_count, batch_size);
    sample_count = std::max<std::size_t>(1U, sample_count);

    if (load_cn_geometry_cache(device_id, props, batch_size, sample_count, result)) {
        ensure_cn_loop_modes(device_id, props, stream, states, sample_count, batch_size,
                             scratchpads, contexts, result);
        ensure_cn_dual_modes(device_id, props, stream, states, sample_count, batch_size,
                             scratchpads, contexts, result);
        return result;
    }

    std::cout << "[GPU " << device_id << "] empirical CryptoNight geometry autotune starting"
              << " | sample=" << sample_count << " | candidates=32,64,128,256\n";
    result[0] = autotune_cn_variant<0>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[1] = autotune_cn_variant<1>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[2] = autotune_cn_variant<2>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[3] = autotune_cn_variant<3>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[4] = autotune_cn_variant<4>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[5] = autotune_cn_variant<5>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    std::cout << "[GPU " << device_id << "] empirical CryptoNight geometry autotune complete\n";

    const bool ttable_valid = validate_aes_ttable_backend(stream);
    std::cout << "[GPU " << device_id << "] shared T-table AES parity validation: "
              << (ttable_valid ? "PASS" : "FAIL - portable AES forced") << '\n';
    benchmark_aes_backend<0>(stream, states, sample_count, scratchpads, contexts, result[0], ttable_valid);
    benchmark_aes_backend<1>(stream, states, sample_count, scratchpads, contexts, result[1], ttable_valid);
    benchmark_aes_backend<2>(stream, states, sample_count, scratchpads, contexts, result[2], ttable_valid);
    benchmark_aes_backend<3>(stream, states, sample_count, scratchpads, contexts, result[3], ttable_valid);
    benchmark_aes_backend<4>(stream, states, sample_count, scratchpads, contexts, result[4], ttable_valid);
    benchmark_aes_backend<5>(stream, states, sample_count, scratchpads, contexts, result[5], ttable_valid);

    ensure_cn_loop_modes(device_id, props, stream, states, sample_count, batch_size,
                         scratchpads, contexts, result);
    ensure_cn_dual_modes(device_id, props, stream, states, sample_count, batch_size,
                         scratchpads, contexts, result);
    print_cn_phase_measurements(device_id, result, sample_count, "fresh-autotune");
    save_cn_geometry_cache(device_id, props, batch_size, sample_count, result);
    return result;
}

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& geometry)
{
    const int setup_threads = geometry.setup_threads;
    const int loop_threads = geometry.loop_threads;
    const int final_threads = geometry.final_threads;
    const int setup_blocks = static_cast<int>((count + setup_threads - 1) / setup_threads);
    const int loop_blocks = static_cast<int>((count + loop_threads - 1) / loop_threads);
    const int final_blocks = static_cast<int>((count + final_threads - 1) / final_threads);
    if (geometry.aes_backend == 1)
        cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight setup launch failed");

    int device_id = -1;
    int loop_mode = 1;
    bool use_dual = false;
    int dual_threads = 32;
    if (cudaGetDevice(&device_id) == cudaSuccess && device_id >= 0 && device_id < kMaxLoopTuneDevices) {
        if (g_cn_loop_ready[device_id]) loop_mode = g_cn_loop_unroll[device_id][VariantIndex];
        if (g_cn_dual_ready[device_id]) {
            use_dual = g_cn_dual_enabled[device_id][VariantIndex];
            dual_threads = g_cn_dual_threads[device_id][VariantIndex];
        }
    } else {
        cudaGetLastError();
    }

    if (use_dual) {
        launch_cn_dual_mode<VariantIndex>(stream, loop_mode, dual_threads, count, scratchpads, contexts);
    } else {
        launch_cn_loop_mode<VariantIndex>(stream, loop_mode, loop_blocks, loop_threads,
                                          count, scratchpads, contexts);
    }
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight loop launch failed");

    if (geometry.aes_backend == 1)
        cryptonight_final_stage<VariantIndex, true><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_final_stage<VariantIndex, false><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight final launch failed");
}

void launch_split_cryptonight(cudaStream_t stream,
                              std::uint8_t* states,
                              std::size_t count,
                              std::uint8_t variant,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              const CnGeometry& geometry)
{
    switch (variant) {
    case 0: launch_split_cryptonight_variant<0>(stream, states, count, scratchpads, contexts, geometry); break;
    case 1: launch_split_cryptonight_variant<1>(stream, states, count, scratchpads, contexts, geometry); break;
    case 2: launch_split_cryptonight_variant<2>(stream, states, count, scratchpads, contexts, geometry); break;
    case 3: launch_split_cryptonight_variant<3>(stream, states, count, scratchpads, contexts, geometry); break;
    case 4: launch_split_cryptonight_variant<4>(stream, states, count, scratchpads, contexts, geometry); break;
    case 5: launch_split_cryptonight_variant<5>(stream, states, count, scratchpads, contexts, geometry); break;
    default: throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
    }
}

__global__ void keccak512_validation_kernel(const std::uint8_t* input, std::size_t length, std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::keccak512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void cubehash512_validation_kernel(const std::uint8_t* input, std::size_t length, std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::cubehash512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void collect_candidates(const std::uint32_t* nonces,
                                   const std::uint8_t* states,
                                   std::size_t count,
                                   const DeviceJob* job,
                                   DeviceCandidate* candidates,
                                   unsigned int* candidate_count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint8_t* hash = states + index * kStateBytes;
    if (!hash_meets_target(hash, job->target_le)) return;
    const unsigned int slot = atomicAdd(candidate_count, 1U);
    if (slot >= kMaxCandidates) return;
    candidates[slot].nonce = nonces[index];
    for (int i = 0; i < 32; ++i) candidates[slot].hash[i] = hash[i];
}

Hash512 run_validation_kernel(int device_id,
                              const std::uint8_t* input,
                              std::size_t length,
                              bool use_keccak)
{
    if (input == nullptr || length == 0)
        throw std::invalid_argument("CUDA core validation input must not be empty");
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), length), "cudaMalloc validation input failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), 64), "cudaMalloc validation output failed");
        check_cuda(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice), "cudaMemcpy validation input failed");
        if (use_keccak) keccak512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        else cubehash512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        check_cuda(cudaGetLastError(), "CUDA core validation launch failed");
        check_cuda(cudaDeviceSynchronize(), "CUDA core validation synchronize failed");
        Hash512 out{};
        check_cuda(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost), "cudaMemcpy validation output failed");
        cudaFree(d_output);
        cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_output) cudaFree(d_output);
        if (d_input) cudaFree(d_input);
        throw;
    }
}

} // namespace
