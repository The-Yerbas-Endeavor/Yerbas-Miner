// Production CryptoNight phase profiling plus generic runtime overlap scheduling.
//
// YERBAS_CUDA_OVERLAP=1 (or "auto") enables a conservative runtime scheduler.
// After the existing lazy parity selectors are warm, the scheduler compares the
// real active batch using single-stream, 33/67 and 50/50 two-stream execution.
// A two-stream layout must beat single-stream by >=7% before promotion. Results
// are cached by GPU runtime/batch/CN variant; no GPU model or architecture rule
// participates in the decision.

namespace {

constexpr int kCnPhaseProfileMaxDevices = 32;
constexpr int kCnSharedKeyMaxThreads = 128;
constexpr float kCnOverlapPromotionRatio = 0.93F;
constexpr int kCnOverlapSchedulerRevision = 1;
static std::array<std::array<std::uint64_t, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_launches{};
static std::array<std::array<double, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_setup_ms{};
static std::array<std::array<double, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_loop_ms{};
static std::array<std::array<double, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_final_ms{};
static std::array<std::array<std::uint64_t, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_samples{};
static std::array<cudaStream_t, kCnPhaseProfileMaxDevices> g_cn_overlap_streams{};
// 0=unknown, 1=single, 33=33/67, 50=50/50.
static std::array<std::array<int, 6>, kCnPhaseProfileMaxDevices> g_cn_overlap_mode{};
static std::array<std::array<bool, 6>, kCnPhaseProfileMaxDevices> g_cn_overlap_reported{};

inline bool cn_phase_diagnostics_enabled()
{
    const char* value = std::getenv("YERBAS_DIAGNOSTICS");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline bool cn_overlap_enabled()
{
    const char* value = std::getenv("YERBAS_CUDA_OVERLAP");
    if (!value || !*value) return false;
    const std::string v(value);
    return v != "0" && v != "off" && v != "false";
}

inline cudaStream_t cn_overlap_stream(int device_id)
{
    if (device_id < 0 || device_id >= kCnPhaseProfileMaxDevices) return nullptr;
    auto& secondary = g_cn_overlap_streams[device_id];
    if (!secondary)
        check_cuda(cudaStreamCreateWithFlags(&secondary, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags CN overlap failed");
    return secondary;
}

template <std::uint8_t VariantIndex, bool UseTTable>
int cn_setup_occupancy_threads(const cudaDeviceProp& props)
{
    const int warp = std::max(32, props.warpSize);
    const int structural_max = std::min(props.maxThreadsPerBlock, kCnSharedKeyMaxThreads);
    return cn_occupancy_block_size(
        cryptonight_setup_stage_cooperative8<VariantIndex, UseTTable>, warp, structural_max);
}

template <std::uint8_t VariantIndex, bool UseTTable>
int cn_final_occupancy_threads(const cudaDeviceProp& props)
{
    const int warp = std::max(32, props.warpSize);
    const int structural_max = std::min(props.maxThreadsPerBlock, kCnSharedKeyMaxThreads);
    return cn_occupancy_block_size(
        cryptonight_final_stage_cooperative8<VariantIndex, UseTTable>, warp, structural_max);
}

template <std::uint8_t VariantIndex>
bool cn_overlap_selectors_ready(int device_id, bool use_ttable)
{
    if (device_id < 0 || device_id >= kCnPhaseProfileMaxDevices) return false;
    if (g_cn_hardened_mode[device_id][VariantIndex] == 0) return false;
    if (use_ttable &&
        (g_cn_word_setup[device_id][VariantIndex] == 0 ||
         g_cn_word_final[device_id][VariantIndex] == 0)) return false;
    return true;
}

template <std::uint8_t VariantIndex>
void launch_cn_partition(cudaStream_t stream,
                         int device_id,
                         std::size_t count,
                         std::uint8_t* states,
                         std::uint8_t* scratchpads,
                         cryptonight::SplitContext* contexts,
                         const CnGeometry& geometry,
                         const cudaDeviceProp& props)
{
    if (count == 0) return;
    const bool use_ttable = geometry.aes_backend == 1;
    const int setup_threads = use_ttable ? kCnSharedKeyMaxThreads
        : cn_setup_occupancy_threads<VariantIndex, false>(props);
    const int final_threads = use_ttable ? kCnSharedKeyMaxThreads
        : cn_final_occupancy_threads<VariantIndex, false>(props);
    const std::size_t total_threads = count * 8U;
    const int setup_blocks = static_cast<int>((total_threads + setup_threads - 1U) / setup_threads);
    const int final_blocks = static_cast<int>((total_threads + final_threads - 1U) / final_threads);

    if (use_ttable) {
        launch_cn_setup_selected<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
    } else {
        cryptonight_setup_stage_cooperative8<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative8 setup launch failed");
    }

    if (use_ttable)
        launch_cn_loop_hardened<VariantIndex, true>(stream, states, count, scratchpads, contexts, geometry);
    else
        launch_cn_loop_hardened<VariantIndex, false>(stream, states, count, scratchpads, contexts, geometry);

    if (use_ttable) {
        launch_cn_final_selected<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
    } else {
        cryptonight_final_stage_cooperative8<VariantIndex, false><<<final_blocks, final_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative8 final launch failed");
    }
}

inline std::size_t cn_overlap_first_count(std::size_t count, int mode)
{
    if (mode != 33 && mode != 50) return count;
    const std::size_t pct = mode == 33 ? 33U : 50U;
    std::size_t first = ((count * pct / 100U) / 32U) * 32U;
    if (first == 0U) first = std::min<std::size_t>(32U, count);
    if (first >= count && count > 32U) first = count - 32U;
    return first;
}

template <std::uint8_t VariantIndex>
void launch_cn_mode(cudaStream_t primary,
                    int device_id,
                    int mode,
                    std::size_t count,
                    std::uint8_t* states,
                    std::uint8_t* scratchpads,
                    cryptonight::SplitContext* contexts,
                    const CnGeometry& geometry,
                    const cudaDeviceProp& props)
{
    if (mode != 33 && mode != 50) {
        launch_cn_partition<VariantIndex>(primary, device_id, count,
                                          states, scratchpads, contexts, geometry, props);
        return;
    }

    const std::size_t first = cn_overlap_first_count(count, mode);
    if (first == 0U || first >= count) {
        launch_cn_partition<VariantIndex>(primary, device_id, count,
                                          states, scratchpads, contexts, geometry, props);
        return;
    }
    const std::size_t second = count - first;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    cudaStream_t secondary = cn_overlap_stream(device_id);

    cudaEvent_t ready{}, done{};
    check_cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cudaEventCreate CN overlap ready failed");
    check_cuda(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cudaEventCreate CN overlap done failed");
    check_cuda(cudaEventRecord(ready, primary), "cudaEventRecord CN overlap ready failed");
    check_cuda(cudaStreamWaitEvent(secondary, ready, 0), "cudaStreamWaitEvent CN overlap ready failed");

    launch_cn_partition<VariantIndex>(primary, device_id, first,
                                      states, scratchpads, contexts, geometry, props);
    launch_cn_partition<VariantIndex>(secondary, device_id, second,
                                      states + first * kStateBytes,
                                      scratchpads + first * cfg.page_size,
                                      contexts + first, geometry, props);

    check_cuda(cudaEventRecord(done, secondary), "cudaEventRecord CN overlap done failed");
    check_cuda(cudaStreamWaitEvent(primary, done, 0), "cudaStreamWaitEvent CN overlap done failed");
    check_cuda(cudaEventDestroy(done), "cudaEventDestroy CN overlap done failed");
    check_cuda(cudaEventDestroy(ready), "cudaEventDestroy CN overlap ready failed");
}

template <std::uint8_t VariantIndex>
std::filesystem::path cn_overlap_cache_path(int device_id,
                                            const cudaDeviceProp& props,
                                            std::size_t count)
{
    int driver = 0, runtime = 0;
    if (cudaDriverGetVersion(&driver) != cudaSuccess || cudaRuntimeGetVersion(&runtime) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, count, driver, runtime);
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    return cn_cache_directory() /
        ("cn-overlap-rev" + std::to_string(kCnOverlapSchedulerRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) +
         "-p" + std::to_string(cfg.page_size) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_overlap_cache(int device_id, const cudaDeviceProp& props, std::size_t count)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_overlap_cache_path<VariantIndex>(device_id, props, count);
    if (path.empty()) return false;
    std::ifstream in(path);
    int revision = 0, mode = 0;
    std::string magic;
    if (!(in >> magic >> revision >> mode) || magic != "YERBAS_CN_OVERLAP" ||
        revision != kCnOverlapSchedulerRevision || (mode != 1 && mode != 33 && mode != 50)) return false;
    g_cn_overlap_mode[device_id][VariantIndex] = mode;
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_overlap_cache(int device_id, const cudaDeviceProp& props, std::size_t count)
{
    try {
        const auto path = cn_overlap_cache_path<VariantIndex>(device_id, props, count);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (out) out << "YERBAS_CN_OVERLAP " << kCnOverlapSchedulerRevision << ' '
                     << g_cn_overlap_mode[device_id][VariantIndex] << '\n';
    } catch (...) {
        // Scheduler cache is optional; mining remains available without it.
    }
}

template <std::uint8_t VariantIndex>
float time_cn_mode(cudaStream_t stream,
                   int device_id,
                   int mode,
                   std::size_t count,
                   std::uint8_t* states,
                   std::uint8_t* scratchpads,
                   cryptonight::SplitContext* contexts,
                   const CnGeometry& geometry,
                   const cudaDeviceProp& props)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate CN overlap timing start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate CN overlap timing stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN overlap timing start failed");
    launch_cn_mode<VariantIndex>(stream, device_id, mode, count, states, scratchpads, contexts, geometry, props);
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN overlap timing stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN overlap timing failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime CN overlap timing failed");
    cudaEventDestroy(stop); cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
bool launch_cn_scheduled_if_ready(cudaStream_t stream,
                                  int device_id,
                                  std::size_t count,
                                  std::uint8_t* states,
                                  std::uint8_t* scratchpads,
                                  cryptonight::SplitContext* contexts,
                                  const CnGeometry& geometry,
                                  const cudaDeviceProp& props)
{
    if (!cn_overlap_enabled() || count < 96U) return false;
    const bool use_ttable = geometry.aes_backend == 1;
    if (!cn_overlap_selectors_ready<VariantIndex>(device_id, use_ttable)) return false;

    int& mode = g_cn_overlap_mode[device_id][VariantIndex];
    if (mode == 0 && !load_cn_overlap_cache<VariantIndex>(device_id, props, count)) {
        // Only state must be preserved. Setup completely reconstructs scratchpad
        // and split context on every candidate execution.
        std::uint8_t* saved_states = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&saved_states), count * kStateBytes),
                   "cudaMalloc CN overlap saved states failed");
        check_cuda(cudaMemcpyAsync(saved_states, states, count * kStateBytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN overlap save states failed");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CN overlap save states failed");

        auto reset = [&]() {
            check_cuda(cudaMemcpyAsync(states, saved_states, count * kStateBytes,
                                       cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpy CN overlap restore states failed");
        };

        reset();
        const float single_ms = time_cn_mode<VariantIndex>(stream, device_id, 1, count,
            states, scratchpads, contexts, geometry, props);
        reset();
        const float split33_ms = time_cn_mode<VariantIndex>(stream, device_id, 33, count,
            states, scratchpads, contexts, geometry, props);
        reset();
        const float split50_ms = time_cn_mode<VariantIndex>(stream, device_id, 50, count,
            states, scratchpads, contexts, geometry, props);

        mode = 1;
        float best_ms = single_ms;
        if (split33_ms < single_ms * kCnOverlapPromotionRatio && split33_ms < best_ms) {
            mode = 33; best_ms = split33_ms;
        }
        if (split50_ms < single_ms * kCnOverlapPromotionRatio && split50_ms < best_ms) {
            mode = 50; best_ms = split50_ms;
        }

        reset();
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CN overlap production reset failed");
        cudaFree(saved_states);
        save_cn_overlap_cache<VariantIndex>(device_id, props, count);

        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA CN scheduler] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | batch=" << count
                  << " | single=" << single_ms << " ms"
                  << " | split33=" << split33_ms << " ms"
                  << " | split50=" << split50_ms << " ms"
                  << " | threshold=7%"
                  << " | production=" << (mode == 1 ? "single" : (mode == 33 ? "33/67" : "50/50"))
                  << std::defaultfloat << '\n';
    }

    launch_cn_mode<VariantIndex>(stream, device_id, mode, count,
                                 states, scratchpads, contexts, geometry, props);
    if (!g_cn_overlap_reported[device_id][VariantIndex]) {
        g_cn_overlap_reported[device_id][VariantIndex] = true;
        const std::size_t first = mode == 1 ? count : cn_overlap_first_count(count, mode);
        std::cout << "[CUDA CN overlap] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | mode=" << (mode == 1 ? "single" : "2-stream")
                  << " | batch=" << count;
        if (mode != 1) std::cout << " | partitions=" << first << '+' << (count - first);
        std::cout << " | generic=yes\n";
    }
    return true;
}

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant_phase_profiled(cudaStream_t stream,
                                                     std::uint8_t* states,
                                                     std::size_t count,
                                                     std::uint8_t* scratchpads,
                                                     cryptonight::SplitContext* contexts,
                                                     const CnGeometry& geometry)
{
    static_assert(VariantIndex < 6, "invalid CryptoNight variant");
    int device_id = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN phase profile failed");
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN phase profile failed");
    const bool use_ttable = geometry.aes_backend == 1;

    // First encounter remains single-stream while lazy parity/timing selectors warm.
    if (launch_cn_scheduled_if_ready<VariantIndex>(stream, device_id, count,
                                                   states, scratchpads, contexts,
                                                   geometry, props)) return;

    const int setup_threads = use_ttable ? kCnSharedKeyMaxThreads
        : cn_setup_occupancy_threads<VariantIndex, false>(props);
    const int final_threads = use_ttable ? kCnSharedKeyMaxThreads
        : cn_final_occupancy_threads<VariantIndex, false>(props);
    const std::size_t total_threads = count * 8U;
    const int setup_blocks = static_cast<int>((total_threads + setup_threads - 1U) / setup_threads);
    const int final_blocks = static_cast<int>((total_threads + final_threads - 1U) / final_threads);

    bool sample = false;
    if (device_id >= 0 && device_id < kCnPhaseProfileMaxDevices && cn_phase_diagnostics_enabled()) {
        auto& launches = g_cn_phase_launches[device_id][VariantIndex];
        ++launches;
        sample = (launches % 8U) == 0U;
    }
    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    if (sample) {
        check_cuda(cudaEventCreateWithFlags(&e0, cudaEventDefault), "cudaEventCreate CN phase setup start failed");
        check_cuda(cudaEventCreateWithFlags(&e1, cudaEventDefault), "cudaEventCreate CN phase setup stop failed");
        check_cuda(cudaEventCreateWithFlags(&e2, cudaEventDefault), "cudaEventCreate CN phase loop stop failed");
        check_cuda(cudaEventCreateWithFlags(&e3, cudaEventDefault), "cudaEventCreate CN phase final stop failed");
        check_cuda(cudaEventRecord(e0, stream), "cudaEventRecord CN phase setup start failed");
    }

    if (use_ttable) {
        launch_cn_setup_selected<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
    } else {
        cryptonight_setup_stage_cooperative8<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative8 setup launch failed");
    }
    if (sample) check_cuda(cudaEventRecord(e1, stream), "cudaEventRecord CN phase setup stop failed");

    if (use_ttable)
        launch_cn_loop_hardened<VariantIndex, true>(stream, states, count, scratchpads, contexts, geometry);
    else
        launch_cn_loop_hardened<VariantIndex, false>(stream, states, count, scratchpads, contexts, geometry);
    if (sample) check_cuda(cudaEventRecord(e2, stream), "cudaEventRecord CN phase loop stop failed");

    if (use_ttable) {
        launch_cn_final_selected<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
    } else {
        cryptonight_final_stage_cooperative8<VariantIndex, false><<<final_blocks, final_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight cooperative8 final launch failed");
    }

    if (!sample) return;
    check_cuda(cudaEventRecord(e3, stream), "cudaEventRecord CN phase final stop failed");
    check_cuda(cudaEventSynchronize(e3), "cudaEventSynchronize CN phase profile failed");
    float setup_ms=0.0F, loop_ms=0.0F, final_ms=0.0F;
    check_cuda(cudaEventElapsedTime(&setup_ms,e0,e1), "cudaEventElapsedTime CN setup failed");
    check_cuda(cudaEventElapsedTime(&loop_ms,e1,e2), "cudaEventElapsedTime CN loop failed");
    check_cuda(cudaEventElapsedTime(&final_ms,e2,e3), "cudaEventElapsedTime CN final failed");
    cudaEventDestroy(e3); cudaEventDestroy(e2); cudaEventDestroy(e1); cudaEventDestroy(e0);

    auto& samples = g_cn_phase_samples[device_id][VariantIndex];
    ++samples;
    g_cn_phase_setup_ms[device_id][VariantIndex] += setup_ms;
    g_cn_phase_loop_ms[device_id][VariantIndex] += loop_ms;
    g_cn_phase_final_ms[device_id][VariantIndex] += final_ms;
    const float total_ms = setup_ms + loop_ms + final_ms;
    const float setup_pct = total_ms > 0.0F ? setup_ms*100.0F/total_ms : 0.0F;
    const float loop_pct = total_ms > 0.0F ? loop_ms*100.0F/total_ms : 0.0F;
    const float final_pct = total_ms > 0.0F ? final_ms*100.0F/total_ms : 0.0F;
    const double avg_setup = g_cn_phase_setup_ms[device_id][VariantIndex] / static_cast<double>(samples);
    const double avg_loop = g_cn_phase_loop_ms[device_id][VariantIndex] / static_cast<double>(samples);
    const double avg_final = g_cn_phase_final_ms[device_id][VariantIndex] / static_cast<double>(samples);
    const int loop_mode = cn_hardened_mode(device_id, VariantIndex);

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN phase] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | batch=" << count
              << " | setup-mode=" << (use_ttable ? cn_word_setup_name(device_id, VariantIndex) : "byte-aes")
              << " final-mode=" << (use_ttable ? cn_word_final_name(device_id, VariantIndex) : "byte-aes")
              << " | setup-threads=" << setup_threads << " final-threads=" << final_threads
              << " | loop-mode=" << cn_residency_mode_name(loop_mode)
              << " | setup=" << setup_ms << " ms (" << std::setprecision(1) << setup_pct << "%)"
              << std::setprecision(3)
              << " | loop=" << loop_ms << " ms (" << std::setprecision(1) << loop_pct << "%)"
              << std::setprecision(3)
              << " | final=" << final_ms << " ms (" << std::setprecision(1) << final_pct << "%)"
              << std::setprecision(3)
              << " | total=" << total_ms << " ms"
              << " | samples=" << samples
              << " | rolling=" << avg_setup << '/' << avg_loop << '/' << avg_final << " ms"
              << std::defaultfloat << '\n';
}

void launch_split_cryptonight_phase_profiled(cudaStream_t stream,
                                             std::uint8_t* states,
                                             std::size_t count,
                                             std::uint8_t variant,
                                             std::uint8_t* scratchpads,
                                             cryptonight::SplitContext* contexts,
                                             const CnGeometry& geometry)
{
    switch (variant) {
    case 0: launch_split_cryptonight_variant_phase_profiled<0>(stream,states,count,scratchpads,contexts,geometry); break;
    case 1: launch_split_cryptonight_variant_phase_profiled<1>(stream,states,count,scratchpads,contexts,geometry); break;
    case 2: launch_split_cryptonight_variant_phase_profiled<2>(stream,states,count,scratchpads,contexts,geometry); break;
    case 3: launch_split_cryptonight_variant_phase_profiled<3>(stream,states,count,scratchpads,contexts,geometry); break;
    case 4: launch_split_cryptonight_variant_phase_profiled<4>(stream,states,count,scratchpads,contexts,geometry); break;
    case 5: launch_split_cryptonight_variant_phase_profiled<5>(stream,states,count,scratchpads,contexts,geometry); break;
    default: throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
    }
}

} // anonymous namespace
