
__global__ void aes_ttable_validation_kernel(unsigned int* mismatches)
{
    __shared__ std::uint32_t aes_tables[1024];
    cryptonight::init_shared_aes_ttables(aes_tables);
    __syncthreads();

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 256U) return;

    alignas(16) std::uint8_t key[32];
    alignas(16) std::uint8_t expanded[240];
    alignas(16) std::uint8_t a[16];
    alignas(16) std::uint8_t b[16];
    #pragma unroll
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>((tid * 29U + i * 17U + 0x53U) & 0xffU);
    #pragma unroll
    for (int i = 0; i < 16; ++i) a[i] = b[i] = static_cast<std::uint8_t>((tid * 11U + i * 23U + 0xa7U) & 0xffU);
    cryptonight::aes256_expand_key(key, expanded);
    cryptonight::aes_pseudo_round(a, expanded);
    cryptonight::aes_pseudo_round_ttable(b, expanded, aes_tables);
    bool same = true;
    #pragma unroll
    for (int i = 0; i < 16; ++i) same = same && (a[i] == b[i]);
    if (!same) atomicAdd(mismatches, 1U);
}

bool validate_aes_ttable_backend(cudaStream_t stream)
{
    unsigned int* d_mismatches = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_mismatches), sizeof(unsigned int)),
               "cudaMalloc AES validation counter failed");
    try {
        check_cuda(cudaMemsetAsync(d_mismatches, 0, sizeof(unsigned int), stream),
                   "cudaMemsetAsync AES validation counter failed");
        aes_ttable_validation_kernel<<<1, 256, 0, stream>>>(d_mismatches);
        check_cuda(cudaGetLastError(), "AES T-table validation launch failed");
        unsigned int mismatches = 0;
        check_cuda(cudaMemcpyAsync(&mismatches, d_mismatches, sizeof(mismatches),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync AES validation result failed");
        check_cuda(cudaStreamSynchronize(stream), "AES T-table validation synchronize failed");
        cudaFree(d_mismatches);
        return mismatches == 0U;
    } catch (...) {
        cudaFree(d_mismatches);
        throw;
    }
}

constexpr int kMaxLoopTuneDevices = 32;
constexpr int kCnLoopCacheRevision = 1;
constexpr int kCnLoopAesCacheRevision = 1;
static std::array<std::array<int, 6>, kMaxLoopTuneDevices> g_cn_loop_unroll{};
static std::array<bool, kMaxLoopTuneDevices> g_cn_loop_ready{};
static std::array<std::array<bool, 6>, kMaxLoopTuneDevices> g_cn_loop_ttable{};
static std::array<bool, kMaxLoopTuneDevices> g_cn_loop_aes_ready{};

template <std::uint8_t VariantIndex, int Unroll, bool UseTTable = false>
__global__ void cryptonight_loop_stage_tuned(std::size_t count,
                                             std::uint8_t* scratchpads,
                                             const cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    cryptonight::split_memory_loop_tuned<VariantIndex, Unroll, UseTTable>(
        scratchpad, contexts[index], UseTTable ? aes_tables : nullptr);
}

template <std::uint8_t VariantIndex>
void launch_cn_loop_backend(cudaStream_t stream,
                            int mode,
                            bool use_ttable,
                            int blocks,
                            int threads,
                            std::size_t count,
                            std::uint8_t* scratchpads,
                            const cryptonight::SplitContext* contexts)
{
    if (use_ttable) {
        if (mode == 4)
            cryptonight_loop_stage_tuned<VariantIndex, 4, true><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
        else if (mode == 2)
            cryptonight_loop_stage_tuned<VariantIndex, 2, true><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
        else
            cryptonight_loop_stage_tuned<VariantIndex, 1, true><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    } else {
        if (mode == 4)
            cryptonight_loop_stage_tuned<VariantIndex, 4, false><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
        else if (mode == 2)
            cryptonight_loop_stage_tuned<VariantIndex, 2, false><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
        else
            cryptonight_loop_stage_tuned<VariantIndex, 1, false><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
    }
}

template <std::uint8_t VariantIndex>
void launch_cn_loop_mode(cudaStream_t stream,
                         int mode,
                         int blocks,
                         int threads,
                         std::size_t count,
                         std::uint8_t* scratchpads,
                         const cryptonight::SplitContext* contexts)
{
    int device_id = -1;
    bool use_ttable = false;
    if (cudaGetDevice(&device_id) == cudaSuccess && device_id >= 0 && device_id < kMaxLoopTuneDevices &&
        g_cn_loop_aes_ready[device_id]) {
        use_ttable = g_cn_loop_ttable[device_id][VariantIndex];
    } else if (device_id < 0 || device_id >= kMaxLoopTuneDevices) {
        cudaGetLastError();
    }
    launch_cn_loop_backend<VariantIndex>(stream, mode, use_ttable, blocks, threads,
                                         count, scratchpads, contexts);
}

std::filesystem::path cn_loop_cache_path(int device_id,
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
    return cn_cache_directory() / ("cn-loop-rev" + std::to_string(kCnLoopCacheRevision) + '-' + key + ".txt");
}

bool load_cn_loop_cache(int device_id,
                        const cudaDeviceProp& props,
                        std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_loop_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision) || magic != "YERBAS_CN_LOOP" || revision != kCnLoopCacheRevision) return false;
    std::array<int, 6> modes{};
    for (std::size_t i = 0; i < modes.size(); ++i) {
        std::size_t variant = 0;
        int mode = 0;
        if (!(in >> variant >> mode) || variant != i || (mode != 1 && mode != 2 && mode != 4)) return false;
        modes[i] = mode;
    }
    g_cn_loop_unroll[device_id] = modes;
    g_cn_loop_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CryptoNight loop cache loaded | " << path.string() << '\n';
    return true;
}

void save_cn_loop_cache(int device_id,
                        const cudaDeviceProp& props,
                        std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || !g_cn_loop_ready[device_id]) return;
    try {
        const auto path = cn_loop_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_LOOP " << kCnLoopCacheRevision << '\n';
        for (std::size_t i = 0; i < 6; ++i) out << i << ' ' << g_cn_loop_unroll[device_id][i] << '\n';
        out.close();
        if (!out) return;
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
        if (!ec) std::cout << "[GPU " << device_id << "] CryptoNight loop cache saved | " << path.string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] CryptoNight loop cache write skipped: " << e.what() << '\n';
    }
}

template <std::uint8_t VariantIndex>
int autotune_cn_loop_variant(cudaStream_t stream,
                             std::uint8_t* states,
                             std::size_t sample_count,
                             std::uint8_t* scratchpads,
                             cryptonight::SplitContext* contexts,
                             const CnGeometry& g)
{
    constexpr std::array<int, 3> modes{{1, 2, 4}};
    float best_ms = 1.0e30F;
    int best_mode = 1;
    const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    const int loop_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.loop_threads) - 1U) /
                                             static_cast<std::size_t>(g.loop_threads));
    for (const int mode : modes) {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes, stream),
                   "cudaMemsetAsync CryptoNight loop autotune states failed");
        if (g.aes_backend == 1)
            cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        else
            cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CryptoNight loop autotune setup prep failed");
        const float ms = time_cuda_phase(stream, [&]() {
            launch_cn_loop_backend<VariantIndex>(stream, mode, false, loop_blocks, g.loop_threads,
                                                 sample_count, scratchpads, contexts);
            check_cuda(cudaGetLastError(), "CryptoNight loop autotune launch failed");
        });
        if (ms < best_ms) { best_ms = ms; best_mode = mode; }
        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA loop test] " << cryptonight::config_value(VariantIndex).name
                  << " | unroll=" << mode << " | " << ms << " ms"
                  << std::defaultfloat << '\n';
    }
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA loop autotune] " << cryptonight::config_value(VariantIndex).name
              << " | selected=unroll" << best_mode << " | " << best_ms << " ms"
              << std::defaultfloat << '\n';
    return best_mode;
}

std::filesystem::path cn_loop_aes_cache_path(int device_id,
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
    return cn_cache_directory() / ("cn-loop-aes-rev" + std::to_string(kCnLoopAesCacheRevision) + '-' + key + ".txt");
}

bool load_cn_loop_aes_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_loop_aes_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision) || magic != "YERBAS_CN_LOOP_AES" || revision != kCnLoopAesCacheRevision)
        return false;
    std::array<bool, 6> selected{};
    for (std::size_t i = 0; i < selected.size(); ++i) {
        std::size_t variant = 0;
        int value = 0;
        if (!(in >> variant >> value) || variant != i || (value != 0 && value != 1)) return false;
        selected[i] = value != 0;
    }
    g_cn_loop_ttable[device_id] = selected;
    g_cn_loop_aes_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CryptoNight loop AES cache loaded | " << path.string() << '\n';
    return true;
}

void save_cn_loop_aes_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || !g_cn_loop_aes_ready[device_id]) return;
    try {
        const auto path = cn_loop_aes_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_LOOP_AES " << kCnLoopAesCacheRevision << '\n';
        for (std::size_t i = 0; i < 6; ++i)
            out << i << ' ' << (g_cn_loop_ttable[device_id][i] ? 1 : 0) << '\n';
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
            std::cout << "[GPU " << device_id << "] CryptoNight loop AES cache saved | " << path.string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] CryptoNight loop AES cache write skipped: " << e.what() << '\n';
    }
}

template <std::uint8_t VariantIndex>
bool validate_cn_loop_aes_variant(cudaStream_t stream,
                                  std::uint8_t* states,
                                  std::uint8_t* scratchpads,
                                  cryptonight::SplitContext* contexts,
                                  const CnGeometry& g,
                                  int unroll)
{
    constexpr std::size_t count = 2U;
    std::array<std::uint8_t, count * kStateBytes> portable{};
    std::array<std::uint8_t, count * kStateBytes> ttable{};
    const int setup_blocks = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    const int loop_blocks = static_cast<int>((count + static_cast<std::size_t>(g.loop_threads) - 1U) /
                                             static_cast<std::size_t>(g.loop_threads));
    const int final_blocks = static_cast<int>((count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                              static_cast<std::size_t>(g.final_threads));
    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x59U + VariantIndex * 19U), count * kStateBytes, stream),
                   "cudaMemsetAsync loop AES parity states failed");
        if (g.aes_backend == 1)
            cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, g.setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "loop AES parity setup failed");
    };
    auto finish = [&]() {
        if (g.aes_backend == 1)
            cryptonight_final_stage<VariantIndex, true><<<final_blocks, g.final_threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_final_stage<VariantIndex, false><<<final_blocks, g.final_threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "loop AES parity final failed");
    };

    prepare();
    launch_cn_loop_backend<VariantIndex>(stream, unroll, false, loop_blocks, g.loop_threads, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "loop AES parity portable launch failed");
    finish();
    check_cuda(cudaMemcpyAsync(portable.data(), states, portable.size(), cudaMemcpyDeviceToHost, stream),
               "loop AES parity portable copy failed");
    check_cuda(cudaStreamSynchronize(stream), "loop AES parity portable synchronize failed");

    prepare();
    launch_cn_loop_backend<VariantIndex>(stream, unroll, true, loop_blocks, g.loop_threads, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "loop AES parity T-table launch failed");
    finish();
    check_cuda(cudaMemcpyAsync(ttable.data(), states, ttable.size(), cudaMemcpyDeviceToHost, stream),
               "loop AES parity T-table copy failed");
    check_cuda(cudaStreamSynchronize(stream), "loop AES parity T-table synchronize failed");
    return portable == ttable;
}

template <std::uint8_t VariantIndex>
void autotune_cn_loop_aes_variant(int device_id,
                                  cudaStream_t stream,
                                  std::uint8_t* states,
                                  std::size_t sample_count,
                                  std::uint8_t* scratchpads,
                                  cryptonight::SplitContext* contexts,
                                  const CnGeometry& g)
{
    const int unroll = g_cn_loop_ready[device_id] ? g_cn_loop_unroll[device_id][VariantIndex] : 1;
    const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    const int loop_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.loop_threads) - 1U) /
                                             static_cast<std::size_t>(g.loop_threads));
    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes, stream),
                   "cudaMemsetAsync loop AES autotune states failed");
        if (g.aes_backend == 1)
            cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        else
            cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "loop AES autotune setup failed");
    };

    prepare();
    const float portable_ms = time_cuda_phase(stream, [&]() {
        launch_cn_loop_backend<VariantIndex>(stream, unroll, false, loop_blocks, g.loop_threads,
                                             sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "loop AES portable benchmark failed");
    });
    prepare();
    const float ttable_ms = time_cuda_phase(stream, [&]() {
        launch_cn_loop_backend<VariantIndex>(stream, unroll, true, loop_blocks, g.loop_threads,
                                             sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "loop AES T-table benchmark failed");
    });
    const bool parity_ok = validate_cn_loop_aes_variant<VariantIndex>(stream, states, scratchpads, contexts, g, unroll);
    const bool use_ttable = parity_ok && ttable_ms < portable_ms * 0.98F;
    g_cn_loop_ttable[device_id][VariantIndex] = use_ttable;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA loop AES autotune] " << cryptonight::config_value(VariantIndex).name
              << " | parity=" << (parity_ok ? "PASS" : "FAIL")
              << " | portable=" << portable_ms << " ms"
              << " | ttable=" << ttable_ms << " ms"
              << " | selected=" << (use_ttable ? "ttable" : "portable")
              << std::defaultfloat << '\n';
}

void ensure_cn_loop_aes_modes(int device_id,
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
    if (load_cn_loop_aes_cache(device_id, props, batch_size)) return;
    std::cout << "[GPU " << device_id << "] empirical CryptoNight loop AES autotune starting | portable vs shared-ttable\n";
    autotune_cn_loop_aes_variant<0>(device_id, stream, states, sample_count, scratchpads, contexts, geometries[0]);
    autotune_cn_loop_aes_variant<1>(device_id, stream, states, sample_count, scratchpads, contexts, geometries[1]);
    autotune_cn_loop_aes_variant<2>(device_id, stream, states, sample_count, scratchpads, contexts, geometries[2]);
    autotune_cn_loop_aes_variant<3>(device_id, stream, states, sample_count, scratchpads, contexts, geometries[3]);
    autotune_cn_loop_aes_variant<4>(device_id, stream, states, sample_count, scratchpads, contexts, geometries[4]);
    autotune_cn_loop_aes_variant<5>(device_id, stream, states, sample_count, scratchpads, contexts, geometries[5]);
    g_cn_loop_aes_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] empirical CryptoNight loop AES autotune complete\n";
    save_cn_loop_aes_cache(device_id, props, batch_size);
}

void ensure_cn_loop_modes(int device_id,
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
    if (!load_cn_loop_cache(device_id, props, batch_size)) {
        std::cout << "[GPU " << device_id << "] empirical CryptoNight loop autotune starting | modes=unroll1,2,4\n";
        g_cn_loop_unroll[device_id][0] = autotune_cn_loop_variant<0>(stream, states, sample_count, scratchpads, contexts, geometries[0]);
        g_cn_loop_unroll[device_id][1] = autotune_cn_loop_variant<1>(stream, states, sample_count, scratchpads, contexts, geometries[1]);
        g_cn_loop_unroll[device_id][2] = autotune_cn_loop_variant<2>(stream, states, sample_count, scratchpads, contexts, geometries[2]);
        g_cn_loop_unroll[device_id][3] = autotune_cn_loop_variant<3>(stream, states, sample_count, scratchpads, contexts, geometries[3]);
        g_cn_loop_unroll[device_id][4] = autotune_cn_loop_variant<4>(stream, states, sample_count, scratchpads, contexts, geometries[4]);
        g_cn_loop_unroll[device_id][5] = autotune_cn_loop_variant<5>(stream, states, sample_count, scratchpads, contexts, geometries[5]);
        g_cn_loop_ready[device_id] = true;
        std::cout << "[GPU " << device_id << "] empirical CryptoNight loop autotune complete\n";
        save_cn_loop_cache(device_id, props, batch_size);
    }
    ensure_cn_loop_aes_modes(device_id, props, stream, states, sample_count, batch_size,
                             scratchpads, contexts, geometries);
}

template <std::uint8_t VariantIndex>
void benchmark_aes_backend(cudaStream_t stream,
                           std::uint8_t* states,
                           std::size_t sample_count,
                           std::uint8_t* scratchpads,
                           cryptonight::SplitContext* contexts,
                           CnGeometry& g,
                           bool ttable_valid)
{
    auto reset_states = [&]() {
        check_cuda(cudaMemsetAsync(states,
                                   static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes,
                                   stream),
                   "cudaMemsetAsync AES backend benchmark states failed");
    };
    const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    const int loop_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.loop_threads) - 1U) /
                                             static_cast<std::size_t>(g.loop_threads));
    const int final_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                              static_cast<std::size_t>(g.final_threads));

    reset_states();
    const float portable_setup = time_cuda_phase(stream, [&]() {
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "portable AES setup benchmark launch failed");
    });
    reset_states();
    cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
    cryptonight_loop_stage<VariantIndex><<<loop_blocks, g.loop_threads, 0, stream>>>(sample_count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "portable AES final benchmark prep failed");
    const float portable_final = time_cuda_phase(stream, [&]() {
        cryptonight_final_stage<VariantIndex, false><<<final_blocks, g.final_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "portable AES final benchmark launch failed");
    });
    g.portable_aes_ms = portable_setup + portable_final;

    if (!ttable_valid) {
        g.ttable_aes_ms = g.portable_aes_ms;
        g.aes_backend = 0;
        return;
    }

    reset_states();
    const float ttable_setup = time_cuda_phase(stream, [&]() {
        cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "T-table AES setup benchmark launch failed");
    });
    reset_states();
    cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
    cryptonight_loop_stage<VariantIndex><<<loop_blocks, g.loop_threads, 0, stream>>>(sample_count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "T-table AES final benchmark prep failed");
    const float ttable_final = time_cuda_phase(stream, [&]() {
        cryptonight_final_stage<VariantIndex, true><<<final_blocks, g.final_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "T-table AES final benchmark launch failed");
    });
    g.ttable_aes_ms = ttable_setup + ttable_final;
    g.aes_backend = (g.ttable_aes_ms < g.portable_aes_ms) ? 1 : 0;

    if (g.aes_backend == 1) {
        g.setup_ms = ttable_setup;
        g.final_ms = ttable_final;
    } else {
        g.setup_ms = portable_setup;
        g.final_ms = portable_final;
    }

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA AES autotune] " << cryptonight::config_value(VariantIndex).name
              << " | portable=" << g.portable_aes_ms << " ms"
              << " | ttable=" << g.ttable_aes_ms << " ms"
              << " | selected=" << (g.aes_backend == 1 ? "ttable" : "portable")
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex>
CnGeometry autotune_cn_variant(cudaStream_t stream,
                               std::uint8_t* states,
                               std::size_t sample_count,
                               std::uint8_t* scratchpads,
                               cryptonight::SplitContext* contexts,
                               int max_threads)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::array<int, 4> candidates{{32, 64, 128, 256}};
    CnGeometry best{};
    float best_setup_ms = 1.0e30F;
    float best_loop_ms = 1.0e30F;
    float best_final_ms = 1.0e30F;
    auto reset_states = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes, stream),
                   "cudaMemsetAsync CryptoNight autotune states failed");
    };
    for (const int threads : candidates) {
        if (threads > max_threads) continue;
        reset_states();
        const int blocks = static_cast<int>((sample_count + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        const float ms = time_cuda_phase(stream, [&]() {
            cryptonight_setup_stage<VariantIndex><<<blocks, threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
            check_cuda(cudaGetLastError(), "CryptoNight autotune setup launch failed");
        });
        if (ms < best_setup_ms) { best_setup_ms = ms; best.setup_threads = threads; }
    }
    for (const int threads : candidates) {
        if (threads > max_threads) continue;
        reset_states();
        const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(best.setup_threads) - 1U) /
                                                  static_cast<std::size_t>(best.setup_threads));
        cryptonight_setup_stage<VariantIndex><<<setup_blocks, best.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CryptoNight autotune loop prep failed");
        const int blocks = static_cast<int>((sample_count + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        const float ms = time_cuda_phase(stream, [&]() {
            cryptonight_loop_stage<VariantIndex><<<blocks, threads, 0, stream>>>(sample_count, scratchpads, contexts);
            check_cuda(cudaGetLastError(), "CryptoNight autotune loop launch failed");
        });
        if (ms < best_loop_ms) { best_loop_ms = ms; best.loop_threads = threads; }
    }