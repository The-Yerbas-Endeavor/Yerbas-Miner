// Final production-batch CN-Fast backend-family selector.
//
// Earlier tuning layers optimize geometry, single-state loop unroll, dual-state
// latency hiding, and cooperative execution independently. This last selector
// compares the useful backend families against each other at the real mining
// batch size, validates the winning candidate against the currently selected
// known-good path, requires a 3% loop-time improvement before switching, and
// caches the decision per GPU / driver / runtime / batch.

namespace {

constexpr int kCnFastFamilyCacheRevision = 1;
constexpr float kCnFastFamilyRequiredSpeedup = 0.97F;

enum class CnFastFamilyMode : int {
    KeepCurrent = 0,
    SingleU1 = 1,
    SingleU2 = 2,
    DualU1T32 = 3,
    DualU1T64 = 4,
    DualU2T32 = 5,
    DualU2T64 = 6,
    Coop4 = 7,
};

struct CnFastFamilyChoice {
    CnFastFamilyMode mode{CnFastFamilyMode::KeepCurrent};
    float ms{0.0F};
};

static std::array<bool, kMaxLoopTuneDevices> g_cn_fast_family_ready{};

bool cn_fast_benchmark_requested()
{
    const char* value = std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

const char* cn_fast_family_name(CnFastFamilyMode mode)
{
    switch (mode) {
    case CnFastFamilyMode::SingleU1: return "single-u1";
    case CnFastFamilyMode::SingleU2: return "single-u2";
    case CnFastFamilyMode::DualU1T32: return "dual-u1-t32";
    case CnFastFamilyMode::DualU1T64: return "dual-u1-t64";
    case CnFastFamilyMode::DualU2T32: return "dual-u2-t32";
    case CnFastFamilyMode::DualU2T64: return "dual-u2-t64";
    case CnFastFamilyMode::Coop4: return "coop4";
    default: return "current";
    }
}

std::filesystem::path cn_fast_family_cache_path(int device_id,
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
    return cn_cache_directory() / ("cn-fast-family-rev" + std::to_string(kCnFastFamilyCacheRevision) + '-' + key + ".txt");
}

void apply_cn_fast_family_choice(int device_id,
                                 CnGeometry& geometry,
                                 CnFastFamilyMode mode)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return;
    constexpr std::size_t V = 2U;

    switch (mode) {
    case CnFastFamilyMode::SingleU1:
        g_cn_coop_enabled[device_id][V] = false;
        g_cn_dual_enabled[device_id][V] = false;
        g_cn_loop_unroll[device_id][V] = 1;
        break;
    case CnFastFamilyMode::SingleU2:
        g_cn_coop_enabled[device_id][V] = false;
        g_cn_dual_enabled[device_id][V] = false;
        g_cn_loop_unroll[device_id][V] = 2;
        break;
    case CnFastFamilyMode::DualU1T32:
    case CnFastFamilyMode::DualU1T64:
        g_cn_coop_enabled[device_id][V] = false;
        g_cn_dual_enabled[device_id][V] = true;
        g_cn_loop_unroll[device_id][V] = 1;
        g_cn_dual_threads[device_id][V] = mode == CnFastFamilyMode::DualU1T64 ? 64 : 32;
        break;
    case CnFastFamilyMode::DualU2T32:
    case CnFastFamilyMode::DualU2T64:
        g_cn_coop_enabled[device_id][V] = false;
        g_cn_dual_enabled[device_id][V] = true;
        g_cn_loop_unroll[device_id][V] = 2;
        g_cn_dual_threads[device_id][V] = mode == CnFastFamilyMode::DualU2T64 ? 64 : 32;
        break;
    case CnFastFamilyMode::Coop4:
        g_cn_coop_enabled[device_id][V] = true;
        g_cn_dual_enabled[device_id][V] = false;
        break;
    default:
        break;
    }

    g_cn_loop_ready[device_id] = true;
    g_cn_dual_ready[device_id] = true;
    g_cn_coop_ready[device_id] = true;
    (void)geometry;
}

CnFastFamilyMode current_cn_fast_family(int device_id)
{
    constexpr std::size_t V = 2U;
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices) return CnFastFamilyMode::KeepCurrent;
    if (g_cn_coop_ready[device_id] && g_cn_coop_enabled[device_id][V]) return CnFastFamilyMode::Coop4;
    if (g_cn_dual_ready[device_id] && g_cn_dual_enabled[device_id][V]) {
        const int u = g_cn_loop_ready[device_id] ? g_cn_loop_unroll[device_id][V] : 1;
        const int t = g_cn_dual_threads[device_id][V];
        if (u == 2) return t == 64 ? CnFastFamilyMode::DualU2T64 : CnFastFamilyMode::DualU2T32;
        return t == 64 ? CnFastFamilyMode::DualU1T64 : CnFastFamilyMode::DualU1T32;
    }
    const int u = g_cn_loop_ready[device_id] ? g_cn_loop_unroll[device_id][V] : 1;
    return u == 2 ? CnFastFamilyMode::SingleU2 : CnFastFamilyMode::SingleU1;
}

void launch_cn_fast_family_loop(cudaStream_t stream,
                                int device_id,
                                CnFastFamilyMode mode,
                                int single_threads,
                                std::size_t count,
                                std::uint8_t* scratchpads,
                                cryptonight::SplitContext* contexts)
{
    constexpr std::uint8_t V = 2;
    if (mode == CnFastFamilyMode::Coop4) {
        int threads = 32;
        if (device_id >= 0 && device_id < kMaxLoopTuneDevices && g_cn_coop_threads[device_id][V] != 0)
            threads = g_cn_coop_threads[device_id][V];
        const int blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        cryptonight_loop_stage_coop4<V><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
        return;
    }

    int unroll = 1;
    int dual_threads = 32;
    bool dual = false;
    switch (mode) {
    case CnFastFamilyMode::SingleU2: unroll = 2; break;
    case CnFastFamilyMode::DualU1T32: dual = true; unroll = 1; dual_threads = 32; break;
    case CnFastFamilyMode::DualU1T64: dual = true; unroll = 1; dual_threads = 64; break;
    case CnFastFamilyMode::DualU2T32: dual = true; unroll = 2; dual_threads = 32; break;
    case CnFastFamilyMode::DualU2T64: dual = true; unroll = 2; dual_threads = 64; break;
    default: break;
    }

    if (dual) {
        launch_cn_dual_mode<V>(stream, unroll, dual_threads, count, scratchpads, contexts);
    } else {
        const int blocks = static_cast<int>((count + static_cast<std::size_t>(single_threads) - 1U) /
                                            static_cast<std::size_t>(single_threads));
        launch_cn_loop_mode<V>(stream, unroll, blocks, single_threads, count, scratchpads, contexts);
    }
}

void prepare_cn_fast_family(cudaStream_t stream,
                            std::uint8_t* states,
                            std::size_t count,
                            std::uint8_t* scratchpads,
                            cryptonight::SplitContext* contexts,
                            const CnGeometry& g)
{
    check_cuda(cudaMemsetAsync(states, 0x69, count * kStateBytes, stream),
               "CN-Fast family tune state reset failed");
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                        static_cast<std::size_t>(g.setup_threads));
    if (g.aes_backend == 1)
        cryptonight_setup_stage<2, true><<<blocks, g.setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage<2, false><<<blocks, g.setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "CN-Fast family tune setup failed");
}

void finalize_cn_fast_family(cudaStream_t stream,
                             std::uint8_t* states,
                             std::size_t count,
                             std::uint8_t* scratchpads,
                             cryptonight::SplitContext* contexts,
                             const CnGeometry& g)
{
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                        static_cast<std::size_t>(g.final_threads));
    if (g.aes_backend == 1)
        cryptonight_final_stage<2, true><<<blocks, g.final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_final_stage<2, false><<<blocks, g.final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "CN-Fast family tune final failed");
}

bool validate_cn_fast_family(cudaStream_t stream,
                             int device_id,
                             CnFastFamilyMode baseline_mode,
                             CnFastFamilyMode candidate_mode,
                             std::uint8_t* states,
                             std::uint8_t* scratchpads,
                             cryptonight::SplitContext* contexts,
                             const CnGeometry& g)
{
    constexpr std::size_t count = 2U;
    std::array<std::uint8_t, count * kStateBytes> baseline{};
    std::array<std::uint8_t, count * kStateBytes> candidate{};

    prepare_cn_fast_family(stream, states, count, scratchpads, contexts, g);
    launch_cn_fast_family_loop(stream, device_id, baseline_mode, g.loop_threads, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "CN-Fast family baseline parity loop failed");
    finalize_cn_fast_family(stream, states, count, scratchpads, contexts, g);
    check_cuda(cudaMemcpyAsync(baseline.data(), states, baseline.size(), cudaMemcpyDeviceToHost, stream),
               "CN-Fast family baseline parity copy failed");
    check_cuda(cudaStreamSynchronize(stream), "CN-Fast family baseline parity sync failed");

    prepare_cn_fast_family(stream, states, count, scratchpads, contexts, g);
    launch_cn_fast_family_loop(stream, device_id, candidate_mode, g.loop_threads, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "CN-Fast family candidate parity loop failed");
    finalize_cn_fast_family(stream, states, count, scratchpads, contexts, g);
    check_cuda(cudaMemcpyAsync(candidate.data(), states, candidate.size(), cudaMemcpyDeviceToHost, stream),
               "CN-Fast family candidate parity copy failed");
    check_cuda(cudaStreamSynchronize(stream), "CN-Fast family candidate parity sync failed");
    return baseline == candidate;
}

bool load_cn_fast_family_cache(int device_id,
                               const cudaDeviceProp& props,
                               std::size_t batch_size,
                               CnGeometry& geometry)
{
    if (cn_fast_benchmark_requested()) return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_fast_family_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    std::string magic;
    int revision = 0;
    int mode_value = 0;
    if (!(in >> magic >> revision >> mode_value) || magic != "YERBAS_CN_FAST_FAMILY" ||
        revision != kCnFastFamilyCacheRevision || mode_value < 0 || mode_value > 7)
        return false;
    const auto mode = static_cast<CnFastFamilyMode>(mode_value);
    if (mode != CnFastFamilyMode::KeepCurrent) apply_cn_fast_family_choice(device_id, geometry, mode);
    g_cn_fast_family_ready[device_id] = true;
    std::cout << "[GPU " << device_id << "] CN-Fast backend cache | selected="
              << cn_fast_family_name(mode) << '\n';
    return true;
}

void save_cn_fast_family_cache(int device_id,
                               const cudaDeviceProp& props,
                               std::size_t batch_size,
                               CnFastFamilyMode mode)
{
    try {
        const auto path = cn_fast_family_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_FAST_FAMILY " << kCnFastFamilyCacheRevision << ' '
            << static_cast<int>(mode) << '\n';
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
        // Cache failure is non-fatal; the next launch simply retunes.
    }
}

void tune_cn_fast_family(int device_id,
                         const cudaDeviceProp& props,
                         cudaStream_t stream,
                         std::uint8_t* states,
                         std::size_t batch_size,
                         std::uint8_t* scratchpads,
                         cryptonight::SplitContext* contexts,
                         CnGeometry& g)
{
    if (device_id < 0 || device_id >= kMaxLoopTuneDevices || batch_size == 0) return;
    if (load_cn_fast_family_cache(device_id, props, batch_size, g)) return;

    const bool benchmark = cn_fast_benchmark_requested();
    const CnFastFamilyMode baseline_mode = current_cn_fast_family(device_id);

    const float setup_ms = time_cuda_phase(stream, [&]() {
        prepare_cn_fast_family(stream, states, batch_size, scratchpads, contexts, g);
    });
    const float baseline_ms = time_cuda_phase(stream, [&]() {
        launch_cn_fast_family_loop(stream, device_id, baseline_mode, g.loop_threads,
                                   batch_size, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast family baseline benchmark failed");
    });
    const float final_ms = time_cuda_phase(stream, [&]() {
        finalize_cn_fast_family(stream, states, batch_size, scratchpads, contexts, g);
    });

    constexpr std::array<CnFastFamilyMode, 7> candidates{{
        CnFastFamilyMode::SingleU1,
        CnFastFamilyMode::SingleU2,
        CnFastFamilyMode::DualU1T32,
        CnFastFamilyMode::DualU1T64,
        CnFastFamilyMode::DualU2T32,
        CnFastFamilyMode::DualU2T64,
        CnFastFamilyMode::Coop4,
    }};

    if (benchmark) {
        const float baseline_hps = baseline_ms > 0.0F
            ? static_cast<float>(batch_size) * 1000.0F / baseline_ms : 0.0F;
        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA CN-Fast benchmark] GPU " << device_id
                  << " | batch=" << batch_size
                  << " | setup=" << setup_ms << " ms"
                  << " | final=" << final_ms << " ms"
                  << " | baseline=" << cn_fast_family_name(baseline_mode)
                  << ' ' << baseline_ms << " ms"
                  << " | loop_hps=" << baseline_hps
                  << std::defaultfloat << '\n';
    }

    CnFastFamilyChoice best{baseline_mode, baseline_ms};
    for (const auto mode : candidates) {
        if ((mode == CnFastFamilyMode::DualU1T64 || mode == CnFastFamilyMode::DualU2T64) &&
            props.maxThreadsPerBlock < 64)
            continue;

        float ms = baseline_ms;
        if (mode != baseline_mode) {
            prepare_cn_fast_family(stream, states, batch_size, scratchpads, contexts, g);
            ms = time_cuda_phase(stream, [&]() {
                launch_cn_fast_family_loop(stream, device_id, mode, g.loop_threads,
                                           batch_size, scratchpads, contexts);
                check_cuda(cudaGetLastError(), "CN-Fast family candidate benchmark failed");
            });
        }
        if (ms < best.ms) best = CnFastFamilyChoice{mode, ms};

        if (benchmark) {
            const float hps = ms > 0.0F ? static_cast<float>(batch_size) * 1000.0F / ms : 0.0F;
            const float gain = ms > 0.0F ? (baseline_ms / ms - 1.0F) * 100.0F : 0.0F;
            std::cout << std::fixed << std::setprecision(3)
                      << "[CUDA CN-Fast candidate] GPU " << device_id
                      << " | mode=" << cn_fast_family_name(mode)
                      << " | loop=" << ms << " ms"
                      << " | loop_hps=" << hps
                      << " | gain=" << gain << "%"
                      << (mode == baseline_mode ? " | baseline" : "")
                      << std::defaultfloat << '\n';
        }
    }

    CnFastFamilyMode selected = baseline_mode;
    bool parity_ok = true;
    if (best.mode != baseline_mode) {
        parity_ok = validate_cn_fast_family(stream, device_id, baseline_mode, best.mode,
                                            states, scratchpads, contexts, g);
        if (parity_ok && best.ms < baseline_ms * kCnFastFamilyRequiredSpeedup)
            selected = best.mode;
    }

    if (selected != baseline_mode) apply_cn_fast_family_choice(device_id, g, selected);
    g_cn_fast_family_ready[device_id] = true;
    save_cn_fast_family_cache(device_id, props, batch_size, selected);

    const float gain = best.ms > 0.0F ? (baseline_ms / best.ms - 1.0F) * 100.0F : 0.0F;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN-Fast autotune] GPU " << device_id
              << " | baseline=" << cn_fast_family_name(baseline_mode) << ' ' << baseline_ms << " ms"
              << " | best=" << cn_fast_family_name(best.mode) << ' ' << best.ms << " ms"
              << " | parity=" << (parity_ok ? "PASS" : "FAIL")
              << " | gain=" << gain << "%"
              << " | selected=" << cn_fast_family_name(selected)
              << std::defaultfloat << '\n';
}

} // anonymous namespace

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    auto result = autotune_cn_geometries_coop(device_id, stream, states, batch_size,
                                              scratchpads, contexts);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties CN-Fast family tuner failed");
    tune_cn_fast_family(device_id, props, stream, states, batch_size,
                        scratchpads, contexts, result[2]);
    return result;
}
