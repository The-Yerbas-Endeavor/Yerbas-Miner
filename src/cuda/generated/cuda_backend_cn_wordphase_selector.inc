// Production-batch CryptoNight setup/final backend selector.
//
// Setup/final are intentionally selected independently from the phase-2 loop
// AES/backend choice. Three architecture-generic implementations compete:
//   1 = cooperative byte AES (production reference)
//   2 = shared-key T-table AES
//   3 = register-word T-table AES
//
// Exact one-hash parity against byte AES is mandatory. Candidates then run on
// the real production batch for three alternating passes and must beat byte AES
// by >=5% on median time before promotion. Selection is cached by CUDA runtime,
// device properties, batch size and CryptoNight variant. No GPU model rule is
// used. Setup tuning snapshots only input states because each setup run fully
// regenerates scratchpad/context. Final tuning snapshots only state/context
// because finalization reads, but does not modify, the phase-2 scratchpad.

namespace {

constexpr int kCnWordPhaseMaxDevices = 32;
constexpr int kCnWordPhasePasses = 3;
constexpr int kCnPhaseBackendCacheRevision = 1;
constexpr float kCnWordPhasePromotionRatio = 0.95F;
constexpr int kCnPhaseBackendByte = 1;
constexpr int kCnPhaseBackendSharedKey = 2;
constexpr int kCnPhaseBackendWord32 = 3;
static std::array<std::array<int, 6>, kCnWordPhaseMaxDevices> g_cn_word_setup{};
static std::array<std::array<int, 6>, kCnWordPhaseMaxDevices> g_cn_word_final{};

template <std::size_t N>
float cn_wordphase_median(std::array<float, N> x)
{
    std::sort(x.begin(), x.end());
    return x[N / 2U];
}

inline const char* cn_phase_backend_name(int mode)
{
    if (mode == kCnPhaseBackendSharedKey) return "sharedkey";
    if (mode == kCnPhaseBackendWord32) return "word32";
    return "byte-aes";
}

template <std::uint8_t VariantIndex>
void cn_launch_setup_phase(cudaStream_t stream, int mode, std::size_t count,
                           std::uint8_t* states, std::uint8_t* scratchpads,
                           cryptonight::SplitContext* contexts)
{
    constexpr int threads = 128;
    const std::size_t total = count * 8U;
    const int blocks = static_cast<int>((total + threads - 1U) / threads);
    if (mode == kCnPhaseBackendWord32) {
        cryptonight_setup_stage_word8<VariantIndex><<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    } else if (mode == kCnPhaseBackendSharedKey) {
        cryptonight_setup_stage_cooperative8_sharedkey<VariantIndex, true><<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    } else {
        cryptonight_setup_stage_cooperative8<VariantIndex, false><<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    }
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN setup backend launch failed");
}

template <std::uint8_t VariantIndex>
void cn_launch_final_phase(cudaStream_t stream, int mode, std::size_t count,
                           std::uint8_t* states, std::uint8_t* scratchpads,
                           cryptonight::SplitContext* contexts)
{
    constexpr int threads = 128;
    const std::size_t total = count * 8U;
    const int blocks = static_cast<int>((total + threads - 1U) / threads);
    if (mode == kCnPhaseBackendWord32) {
        cryptonight_final_stage_word8<VariantIndex><<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    } else if (mode == kCnPhaseBackendSharedKey) {
        cryptonight_final_stage_cooperative8_sharedkey<VariantIndex, true><<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    } else {
        cryptonight_final_stage_cooperative8<VariantIndex, false><<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    }
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN final backend launch failed");
}

template <typename Launch>
float cn_wordphase_time(cudaStream_t stream, Launch&& launch)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate CN phase backend start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate CN phase backend stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN phase backend start failed");
    launch();
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN phase backend stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN phase backend failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime CN phase backend failed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
std::filesystem::path cn_phase_backend_cache_path(int device_id,
                                                  const cudaDeviceProp& props,
                                                  std::size_t count,
                                                  const char* phase)
{
    int driver = 0, runtime = 0;
    if (cudaDriverGetVersion(&driver) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, count, driver, runtime);
    return cn_cache_directory() /
        ("cn-phase-backend-rev" + std::to_string(kCnPhaseBackendCacheRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) +
         '-' + phase + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_phase_backend_cache(int device_id,
                                 const cudaDeviceProp& props,
                                 std::size_t count,
                                 const char* phase,
                                 int& mode)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_phase_backend_cache_path<VariantIndex>(device_id, props, count, phase);
    if (path.empty()) return false;
    std::ifstream in(path);
    std::string magic, cached_phase;
    int revision = 0, variant = -1, cached_mode = 0;
    if (!(in >> magic >> revision >> variant >> cached_phase >> cached_mode) ||
        magic != "YERBAS_CN_PHASE_BACKEND" ||
        revision != kCnPhaseBackendCacheRevision ||
        variant != static_cast<int>(VariantIndex) || cached_phase != phase ||
        (cached_mode != kCnPhaseBackendByte &&
         cached_mode != kCnPhaseBackendSharedKey &&
         cached_mode != kCnPhaseBackendWord32))
        return false;
    mode = cached_mode;
    std::cout << "[GPU " << device_id << "] CryptoNight phase backend cache loaded | "
              << cryptonight::config_value(VariantIndex).name
              << " | phase=" << phase << " | batch=" << count
              << " | production=" << cn_phase_backend_name(mode) << '\n';
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_phase_backend_cache(int device_id,
                                 const cudaDeviceProp& props,
                                 std::size_t count,
                                 const char* phase,
                                 int mode)
{
    try {
        const auto path = cn_phase_backend_cache_path<VariantIndex>(device_id, props, count, phase);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_PHASE_BACKEND " << kCnPhaseBackendCacheRevision << ' '
            << static_cast<int>(VariantIndex) << ' ' << phase << ' ' << mode << '\n';
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
bool cn_validate_setup_backend(cudaStream_t stream,
                               int candidate_mode,
                               const std::uint8_t* states)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t *s0=nullptr,*s1=nullptr,*p0=nullptr,*p1=nullptr;
    cryptonight::SplitContext *c0=nullptr,*c1=nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s0), kStateBytes), "cudaMalloc setup parity state0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s1), kStateBytes), "cudaMalloc setup parity state1 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p0), cfg.page_size), "cudaMalloc setup parity pad0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p1), cfg.page_size), "cudaMalloc setup parity pad1 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c0), sizeof(*c0)), "cudaMalloc setup parity ctx0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c1), sizeof(*c1)), "cudaMalloc setup parity ctx1 failed");
    check_cuda(cudaMemcpyAsync(s0, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy setup parity state0 failed");
    check_cuda(cudaMemcpyAsync(s1, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy setup parity state1 failed");
    cn_launch_setup_phase<VariantIndex>(stream, kCnPhaseBackendByte, 1, s0, p0, c0);
    cn_launch_setup_phase<VariantIndex>(stream, candidate_mode, 1, s1, p1, c1);
    std::vector<std::uint8_t> hp0(cfg.page_size), hp1(cfg.page_size);
    cryptonight::SplitContext hc0{}, hc1{};
    check_cuda(cudaMemcpyAsync(hp0.data(), p0, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity pad0 result failed");
    check_cuda(cudaMemcpyAsync(hp1.data(), p1, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity pad1 result failed");
    check_cuda(cudaMemcpyAsync(&hc0, c0, sizeof(hc0), cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity ctx0 result failed");
    check_cuda(cudaMemcpyAsync(&hc1, c1, sizeof(hc1), cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity ctx1 result failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize setup parity failed");
    cudaFree(c1); cudaFree(c0); cudaFree(p1); cudaFree(p0); cudaFree(s1); cudaFree(s0);
    return hp0 == hp1 && std::memcmp(&hc0, &hc1, sizeof(hc0)) == 0;
}

template <std::uint8_t VariantIndex>
bool cn_validate_final_backend(cudaStream_t stream,
                               int candidate_mode,
                               const std::uint8_t* states,
                               const std::uint8_t* scratchpads,
                               const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t *s0=nullptr,*s1=nullptr,*p=nullptr;
    cryptonight::SplitContext *c0=nullptr,*c1=nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s0), kStateBytes), "cudaMalloc final parity state0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s1), kStateBytes), "cudaMalloc final parity state1 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p), cfg.page_size), "cudaMalloc final parity pad failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c0), sizeof(*c0)), "cudaMalloc final parity ctx0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c1), sizeof(*c1)), "cudaMalloc final parity ctx1 failed");
    check_cuda(cudaMemcpyAsync(s0, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity state0 failed");
    check_cuda(cudaMemcpyAsync(s1, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity state1 failed");
    check_cuda(cudaMemcpyAsync(p, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity pad failed");
    check_cuda(cudaMemcpyAsync(c0, contexts, sizeof(*c0), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity ctx0 failed");
    check_cuda(cudaMemcpyAsync(c1, contexts, sizeof(*c1), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity ctx1 failed");
    cn_launch_final_phase<VariantIndex>(stream, kCnPhaseBackendByte, 1, s0, p, c0);
    cn_launch_final_phase<VariantIndex>(stream, candidate_mode, 1, s1, p, c1);
    std::array<std::uint8_t, kStateBytes> hs0{}, hs1{};
    cryptonight::SplitContext hc0{}, hc1{};
    check_cuda(cudaMemcpyAsync(hs0.data(), s0, kStateBytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity state0 result failed");
    check_cuda(cudaMemcpyAsync(hs1.data(), s1, kStateBytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity state1 result failed");
    check_cuda(cudaMemcpyAsync(&hc0, c0, sizeof(hc0), cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity ctx0 result failed");
    check_cuda(cudaMemcpyAsync(&hc1, c1, sizeof(hc1), cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity ctx1 result failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize final parity failed");
    cudaFree(c1); cudaFree(c0); cudaFree(p); cudaFree(s1); cudaFree(s0);
    return hs0 == hs1 && std::memcmp(&hc0, &hc1, sizeof(hc0)) == 0;
}

template <std::uint8_t VariantIndex>
void cn_select_setup_word(cudaStream_t stream, int device_id, std::size_t count,
                          std::uint8_t* states, std::uint8_t* scratchpads,
                          cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices ||
        g_cn_word_setup[device_id][VariantIndex] != 0) return;
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties setup backend failed");
    int& selected = g_cn_word_setup[device_id][VariantIndex];
    if (load_cn_phase_backend_cache<VariantIndex>(device_id, props, count, "setup", selected)) return;

    const bool shared_ok = cn_validate_setup_backend<VariantIndex>(stream, kCnPhaseBackendSharedKey, states);
    const bool word_ok = cn_validate_setup_backend<VariantIndex>(stream, kCnPhaseBackendWord32, states);

    std::uint8_t* saved_states = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&saved_states), count * kStateBytes),
               "cudaMalloc setup backend saved states failed");
    check_cuda(cudaMemcpyAsync(saved_states, states, count * kStateBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy setup backend save states failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize setup backend save failed");
    auto reset = [&]() {
        check_cuda(cudaMemcpyAsync(states, saved_states, count * kStateBytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy setup backend restore states failed");
    };
    auto timed = [&](int mode) {
        reset();
        return cn_wordphase_time(stream, [&]() {
            cn_launch_setup_phase<VariantIndex>(stream, mode, count, states, scratchpads, contexts);
        });
    };

    // Warm each valid implementation once before measured alternating passes.
    (void)timed(kCnPhaseBackendByte);
    if (shared_ok) (void)timed(kCnPhaseBackendSharedKey);
    if (word_ok) (void)timed(kCnPhaseBackendWord32);

    std::array<float, kCnWordPhasePasses> byte_times{};
    std::array<float, kCnWordPhasePasses> shared_times{};
    std::array<float, kCnWordPhasePasses> word_times{};
    shared_times.fill(-1.0F); word_times.fill(-1.0F);
    for (int pass = 0; pass < kCnWordPhasePasses; ++pass) {
        if ((pass & 1) == 0) {
            byte_times[pass] = timed(kCnPhaseBackendByte);
            if (shared_ok) shared_times[pass] = timed(kCnPhaseBackendSharedKey);
            if (word_ok) word_times[pass] = timed(kCnPhaseBackendWord32);
        } else {
            if (word_ok) word_times[pass] = timed(kCnPhaseBackendWord32);
            if (shared_ok) shared_times[pass] = timed(kCnPhaseBackendSharedKey);
            byte_times[pass] = timed(kCnPhaseBackendByte);
        }
    }
    const float byte_ms = cn_wordphase_median(byte_times);
    const float shared_ms = shared_ok ? cn_wordphase_median(shared_times) : -1.0F;
    const float word_ms = word_ok ? cn_wordphase_median(word_times) : -1.0F;
    selected = kCnPhaseBackendByte;
    float best_ms = byte_ms;
    if (shared_ok && shared_ms < byte_ms * kCnWordPhasePromotionRatio && shared_ms < best_ms) {
        selected = kCnPhaseBackendSharedKey; best_ms = shared_ms;
    }
    if (word_ok && word_ms < byte_ms * kCnWordPhasePromotionRatio && word_ms < best_ms) {
        selected = kCnPhaseBackendWord32; best_ms = word_ms;
    }
    save_cn_phase_backend_cache<VariantIndex>(device_id, props, count, "setup", selected);
    reset();
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize setup backend production reset failed");
    cudaFree(saved_states);

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN phase backend] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | phase=setup | batch=" << count
              << " | byte=" << byte_ms << " ms"
              << " | sharedkey=" << shared_ms << " ms parity=" << (shared_ok ? "PASS" : "FAIL")
              << " | word32=" << word_ms << " ms parity=" << (word_ok ? "PASS" : "FAIL")
              << " | threshold=5% | production=" << cn_phase_backend_name(selected)
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex>
void cn_select_final_word(cudaStream_t stream, int device_id, std::size_t count,
                          std::uint8_t* states, std::uint8_t* scratchpads,
                          cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices ||
        g_cn_word_final[device_id][VariantIndex] != 0) return;
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties final backend failed");
    int& selected = g_cn_word_final[device_id][VariantIndex];
    if (load_cn_phase_backend_cache<VariantIndex>(device_id, props, count, "final", selected)) return;

    const bool shared_ok = cn_validate_final_backend<VariantIndex>(
        stream, kCnPhaseBackendSharedKey, states, scratchpads, contexts);
    const bool word_ok = cn_validate_final_backend<VariantIndex>(
        stream, kCnPhaseBackendWord32, states, scratchpads, contexts);

    std::uint8_t* saved_states = nullptr;
    cryptonight::SplitContext* saved_contexts = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&saved_states), count * kStateBytes),
               "cudaMalloc final backend saved states failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&saved_contexts), count * sizeof(*saved_contexts)),
               "cudaMalloc final backend saved contexts failed");
    check_cuda(cudaMemcpyAsync(saved_states, states, count * kStateBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy final backend save states failed");
    check_cuda(cudaMemcpyAsync(saved_contexts, contexts, count * sizeof(*saved_contexts),
                               cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy final backend save contexts failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize final backend save failed");
    auto reset = [&]() {
        check_cuda(cudaMemcpyAsync(states, saved_states, count * kStateBytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy final backend restore states failed");
        check_cuda(cudaMemcpyAsync(contexts, saved_contexts, count * sizeof(*saved_contexts),
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy final backend restore contexts failed");
    };
    auto timed = [&](int mode) {
        reset();
        return cn_wordphase_time(stream, [&]() {
            cn_launch_final_phase<VariantIndex>(stream, mode, count, states, scratchpads, contexts);
        });
    };

    (void)timed(kCnPhaseBackendByte);
    if (shared_ok) (void)timed(kCnPhaseBackendSharedKey);
    if (word_ok) (void)timed(kCnPhaseBackendWord32);

    std::array<float, kCnWordPhasePasses> byte_times{};
    std::array<float, kCnWordPhasePasses> shared_times{};
    std::array<float, kCnWordPhasePasses> word_times{};
    shared_times.fill(-1.0F); word_times.fill(-1.0F);
    for (int pass = 0; pass < kCnWordPhasePasses; ++pass) {
        if ((pass & 1) == 0) {
            byte_times[pass] = timed(kCnPhaseBackendByte);
            if (shared_ok) shared_times[pass] = timed(kCnPhaseBackendSharedKey);
            if (word_ok) word_times[pass] = timed(kCnPhaseBackendWord32);
        } else {
            if (word_ok) word_times[pass] = timed(kCnPhaseBackendWord32);
            if (shared_ok) shared_times[pass] = timed(kCnPhaseBackendSharedKey);
            byte_times[pass] = timed(kCnPhaseBackendByte);
        }
    }
    const float byte_ms = cn_wordphase_median(byte_times);
    const float shared_ms = shared_ok ? cn_wordphase_median(shared_times) : -1.0F;
    const float word_ms = word_ok ? cn_wordphase_median(word_times) : -1.0F;
    selected = kCnPhaseBackendByte;
    float best_ms = byte_ms;
    if (shared_ok && shared_ms < byte_ms * kCnWordPhasePromotionRatio && shared_ms < best_ms) {
        selected = kCnPhaseBackendSharedKey; best_ms = shared_ms;
    }
    if (word_ok && word_ms < byte_ms * kCnWordPhasePromotionRatio && word_ms < best_ms) {
        selected = kCnPhaseBackendWord32; best_ms = word_ms;
    }
    save_cn_phase_backend_cache<VariantIndex>(device_id, props, count, "final", selected);
    reset();
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize final backend production reset failed");
    cudaFree(saved_contexts);
    cudaFree(saved_states);

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN phase backend] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | phase=final | batch=" << count
              << " | byte=" << byte_ms << " ms"
              << " | sharedkey=" << shared_ms << " ms parity=" << (shared_ok ? "PASS" : "FAIL")
              << " | word32=" << word_ms << " ms parity=" << (word_ok ? "PASS" : "FAIL")
              << " | threshold=5% | production=" << cn_phase_backend_name(selected)
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex>
void launch_cn_setup_selected(cudaStream_t stream,int device_id,std::size_t count,
                              std::uint8_t* states,std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices) {
        cn_launch_setup_phase<VariantIndex>(stream, kCnPhaseBackendByte, count, states, scratchpads, contexts);
        return;
    }
    cn_select_setup_word<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
    cn_launch_setup_phase<VariantIndex>(stream, g_cn_word_setup[device_id][VariantIndex],
                                        count, states, scratchpads, contexts);
}

template <std::uint8_t VariantIndex>
void launch_cn_final_selected(cudaStream_t stream,int device_id,std::size_t count,
                              std::uint8_t* states,std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices) {
        cn_launch_final_phase<VariantIndex>(stream, kCnPhaseBackendByte, count, states, scratchpads, contexts);
        return;
    }
    cn_select_final_word<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
    cn_launch_final_phase<VariantIndex>(stream, g_cn_word_final[device_id][VariantIndex],
                                        count, states, scratchpads, contexts);
}

inline const char* cn_word_setup_name(int device_id,std::uint8_t variant)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices || variant >= 6U) return "byte-aes";
    return cn_phase_backend_name(g_cn_word_setup[device_id][variant]);
}
inline const char* cn_word_final_name(int device_id,std::uint8_t variant)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices || variant >= 6U) return "byte-aes";
    return cn_phase_backend_name(g_cn_word_final[device_id][variant]);
}

} // anonymous namespace
