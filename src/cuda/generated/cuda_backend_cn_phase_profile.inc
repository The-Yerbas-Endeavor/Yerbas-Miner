// Sparse production CryptoNight phase profiling plus optional generic overlap.
//
// YERBAS_CUDA_OVERLAP=1 enables two disjoint nonce partitions for each
// CryptoNight stage. Each partition executes the canonical setup -> loop -> final
// dependency chain on its own stream, then the primary stream joins before the
// next GhostRider stage. No GPU model or compute-capability rule is used.
// Lazy kernel selectors always warm on the single-stream path before overlap.

namespace {

constexpr int kCnPhaseProfileMaxDevices = 32;
constexpr int kCnSharedKeyMaxThreads = 128;
static std::array<std::array<std::uint64_t, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_launches{};
static std::array<std::array<double, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_setup_ms{};
static std::array<std::array<double, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_loop_ms{};
static std::array<std::array<double, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_final_ms{};
static std::array<std::array<std::uint64_t, 6>, kCnPhaseProfileMaxDevices> g_cn_phase_samples{};
static std::array<cudaStream_t, kCnPhaseProfileMaxDevices> g_cn_overlap_streams{};
static std::array<std::array<bool, 6>, kCnPhaseProfileMaxDevices> g_cn_overlap_reported{};

inline bool cn_phase_diagnostics_enabled()
{
    const char* value = std::getenv("YERBAS_DIAGNOSTICS");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline bool cn_overlap_enabled()
{
    const char* value = std::getenv("YERBAS_CUDA_OVERLAP");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
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
        cryptonight_setup_stage_cooperative8<VariantIndex, UseTTable>,
        warp,
        structural_max);
}

template <std::uint8_t VariantIndex, bool UseTTable>
int cn_final_occupancy_threads(const cudaDeviceProp& props)
{
    const int warp = std::max(32, props.warpSize);
    const int structural_max = std::min(props.maxThreadsPerBlock, kCnSharedKeyMaxThreads);
    return cn_occupancy_block_size(
        cryptonight_final_stage_cooperative8<VariantIndex, UseTTable>,
        warp,
        structural_max);
}

template <std::uint8_t VariantIndex>
bool cn_overlap_selectors_ready(int device_id, bool use_ttable)
{
    if (device_id < 0 || device_id >= kCnPhaseProfileMaxDevices) return false;
    if (g_cn_hardened_mode[device_id][VariantIndex] == 0) return false;
    if (use_ttable &&
        (g_cn_word_setup[device_id][VariantIndex] == 0 ||
         g_cn_word_final[device_id][VariantIndex] == 0))
        return false;
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
    const int setup_blocks = static_cast<int>(
        (total_threads + static_cast<std::size_t>(setup_threads) - 1U) /
        static_cast<std::size_t>(setup_threads));
    const int final_blocks = static_cast<int>(
        (total_threads + static_cast<std::size_t>(final_threads) - 1U) /
        static_cast<std::size_t>(final_threads));

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

template <std::uint8_t VariantIndex>
bool launch_cn_overlap_if_ready(cudaStream_t primary,
                                int device_id,
                                std::size_t count,
                                std::uint8_t* states,
                                std::uint8_t* scratchpads,
                                cryptonight::SplitContext* contexts,
                                const CnGeometry& geometry,
                                const cudaDeviceProp& props)
{
    if (!cn_overlap_enabled() || count < 64U) return false;
    const bool use_ttable = geometry.aes_backend == 1;
    if (!cn_overlap_selectors_ready<VariantIndex>(device_id, use_ttable)) return false;

    cudaStream_t secondary = cn_overlap_stream(device_id);
    if (!secondary) return false;

    // Keep subgroup-friendly partition boundaries without assuming any GPU model.
    std::size_t first = (count / 2U / 32U) * 32U;
    if (first == 0 || first >= count) return false;
    const std::size_t second = count - first;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);

    std::uint8_t* states2 = states + first * kStateBytes;
    std::uint8_t* scratch2 = scratchpads + first * cfg.page_size;
    cryptonight::SplitContext* contexts2 = contexts + first;

    cudaEvent_t ready{}, secondary_done{};
    check_cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
               "cudaEventCreate CN overlap ready failed");
    check_cuda(cudaEventCreateWithFlags(&secondary_done, cudaEventDisableTiming),
               "cudaEventCreate CN overlap done failed");
    check_cuda(cudaEventRecord(ready, primary), "cudaEventRecord CN overlap ready failed");
    check_cuda(cudaStreamWaitEvent(secondary, ready, 0), "cudaStreamWaitEvent CN overlap ready failed");

    launch_cn_partition<VariantIndex>(primary, device_id, first,
                                      states, scratchpads, contexts, geometry, props);
    launch_cn_partition<VariantIndex>(secondary, device_id, second,
                                      states2, scratch2, contexts2, geometry, props);

    check_cuda(cudaEventRecord(secondary_done, secondary), "cudaEventRecord CN overlap done failed");
    check_cuda(cudaStreamWaitEvent(primary, secondary_done, 0), "cudaStreamWaitEvent CN overlap done failed");
    check_cuda(cudaEventDestroy(secondary_done), "cudaEventDestroy CN overlap done failed");
    check_cuda(cudaEventDestroy(ready), "cudaEventDestroy CN overlap ready failed");

    if (!g_cn_overlap_reported[device_id][VariantIndex]) {
        g_cn_overlap_reported[device_id][VariantIndex] = true;
        std::cout << "[CUDA CN overlap] GPU " << device_id
                  << " | " << cfg.name
                  << " | mode=2-stream"
                  << " | batch=" << count
                  << " | partitions=" << first << '+' << second
                  << " | generic=yes\n";
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

    // First encounter of a variant remains single-stream so lazy parity/timing
    // selectors cannot race between streams. Later encounters may overlap.
    if (launch_cn_overlap_if_ready<VariantIndex>(stream, device_id, count,
                                                 states, scratchpads, contexts,
                                                 geometry, props)) {
        return;
    }

    const int setup_threads = use_ttable ? kCnSharedKeyMaxThreads
        : cn_setup_occupancy_threads<VariantIndex, false>(props);
    const int final_threads = use_ttable ? kCnSharedKeyMaxThreads
        : cn_final_occupancy_threads<VariantIndex, false>(props);
    const std::size_t phase_total_threads = count * 8U;
    const int setup_blocks = static_cast<int>(
        (phase_total_threads + static_cast<std::size_t>(setup_threads) - 1U) /
        static_cast<std::size_t>(setup_threads));
    const int final_blocks = static_cast<int>(
        (phase_total_threads + static_cast<std::size_t>(final_threads) - 1U) /
        static_cast<std::size_t>(final_threads));

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
    float setup_ms = 0.0F, loop_ms = 0.0F, final_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&setup_ms, e0, e1), "cudaEventElapsedTime CN setup failed");
    check_cuda(cudaEventElapsedTime(&loop_ms, e1, e2), "cudaEventElapsedTime CN loop failed");
    check_cuda(cudaEventElapsedTime(&final_ms, e2, e3), "cudaEventElapsedTime CN final failed");
    cudaEventDestroy(e3); cudaEventDestroy(e2); cudaEventDestroy(e1); cudaEventDestroy(e0);

    auto& samples = g_cn_phase_samples[device_id][VariantIndex];
    ++samples;
    g_cn_phase_setup_ms[device_id][VariantIndex] += static_cast<double>(setup_ms);
    g_cn_phase_loop_ms[device_id][VariantIndex] += static_cast<double>(loop_ms);
    g_cn_phase_final_ms[device_id][VariantIndex] += static_cast<double>(final_ms);

    const float total_ms = setup_ms + loop_ms + final_ms;
    const float setup_pct = total_ms > 0.0F ? setup_ms * 100.0F / total_ms : 0.0F;
    const float loop_pct = total_ms > 0.0F ? loop_ms * 100.0F / total_ms : 0.0F;
    const float final_pct = total_ms > 0.0F ? final_ms * 100.0F / total_ms : 0.0F;
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
              << " | setup-threads=" << setup_threads
              << " final-threads=" << final_threads
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
    case 0: launch_split_cryptonight_variant_phase_profiled<0>(stream, states, count, scratchpads, contexts, geometry); break;
    case 1: launch_split_cryptonight_variant_phase_profiled<1>(stream, states, count, scratchpads, contexts, geometry); break;
    case 2: launch_split_cryptonight_variant_phase_profiled<2>(stream, states, count, scratchpads, contexts, geometry); break;
    case 3: launch_split_cryptonight_variant_phase_profiled<3>(stream, states, count, scratchpads, contexts, geometry); break;
    case 4: launch_split_cryptonight_variant_phase_profiled<4>(stream, states, count, scratchpads, contexts, geometry); break;
    case 5: launch_split_cryptonight_variant_phase_profiled<5>(stream, states, count, scratchpads, contexts, geometry); break;
    default: throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
    }
}

} // anonymous namespace
