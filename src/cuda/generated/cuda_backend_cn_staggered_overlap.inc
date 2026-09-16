// Generic staggered CryptoNight overlap scheduler.
//
// Rev8 restores fully position-balanced timing for the proven coarse scheduler
// families and adds partition-specific phase-2 geometry. Each selected split may
// use a different block size for partition A and B. Geometry is promoted only
// when an alternating full-schedule confirmation beats the baseline pair by at
// least 1%. Runtime timing decides every choice; there are no GPU model rules.
//
// Production remains cache-first. With YERBAS_CUDA_OVERLAP unset or "auto", a
// previously proven rev8 cache is used automatically, but a missing cache never
// pauses mining to benchmark. Explicit overlap tuning is still required to
// create a new cache.

namespace {

constexpr float kCnStaggerPromotionRatio = 0.97F;
constexpr float kCnStaggerGeometryPromotionRatio = 0.99F;
constexpr int kCnStaggerSchedulerRevision = 8;
constexpr std::array<int, 11> kCnStaggerModes{{
    1, 25, 33, 40, 50, 60, 67, 75, 125, 150, 175
}};
constexpr int kCnStaggerTunePasses = static_cast<int>(kCnStaggerModes.size());
constexpr int kCnStaggerGeometryPasses = 3;
constexpr int kCnStaggerGeometryConfirmPasses = 5;
constexpr int kCnStaggerGeometryMaxCandidates = 8;

static std::array<std::array<int, 6>, kCnPhaseProfileMaxDevices> g_cn_stagger_mode{};
static std::array<std::array<int, 6>, kCnPhaseProfileMaxDevices> g_cn_stagger_threads_a{};
static std::array<std::array<int, 6>, kCnPhaseProfileMaxDevices> g_cn_stagger_threads_b{};
static std::array<std::array<bool, 6>, kCnPhaseProfileMaxDevices> g_cn_stagger_reported{};

inline bool cn_stagger_disabled()
{
    const char* value = std::getenv("YERBAS_CUDA_OVERLAP");
    if (!value || !*value) return false;
    const std::string v(value);
    return v == "0" || v == "off" || v == "false" || v == "no";
}

inline bool cn_stagger_tuning_requested()
{
    const char* value = std::getenv("YERBAS_CUDA_OVERLAP");
    if (!value || !*value) return false;
    const std::string v(value);
    return v == "1" || v == "on" || v == "true" || v == "yes" ||
           v == "tune" || v == "full" || v == "benchmark";
}

inline bool cn_stagger_loop_overlap(int mode)
{
    return mode >= 100;
}

inline int cn_stagger_ratio(int mode)
{
    return cn_stagger_loop_overlap(mode) ? mode - 100 : mode;
}

inline bool cn_stagger_split_mode(int mode)
{
    const int ratio = cn_stagger_ratio(mode);
    return ratio >= 20 && ratio <= 80;
}

inline std::size_t cn_stagger_first_count(std::size_t count, int mode)
{
    if (!cn_stagger_split_mode(mode)) return count;
    const int ratio = cn_stagger_ratio(mode);
    std::size_t first =
        ((count * static_cast<std::size_t>(ratio) / 100U) / 32U) * 32U;
    if (first == 0U) first = std::min<std::size_t>(32U, count);
    if (first >= count && count > 32U) first = count - 32U;
    return first;
}

template <std::size_t N>
float cn_stagger_median(std::array<float, N> values)
{
    static_assert(N > 0U, "median requires samples");
    for (std::size_t i = 1; i < values.size(); ++i) {
        const float value = values[i];
        std::size_t j = i;
        while (j > 0U && values[j - 1U] > value) {
            values[j] = values[j - 1U];
            --j;
        }
        values[j] = value;
    }
    if constexpr ((N & 1U) != 0U) {
        return values[N / 2U];
    } else {
        return (values[(N / 2U) - 1U] + values[N / 2U]) * 0.5F;
    }
}

template <std::uint8_t VariantIndex>
void launch_cn_setup_partition(cudaStream_t stream,
                               int device_id,
                               std::size_t count,
                               std::uint8_t* states,
                               std::uint8_t* scratchpads,
                               cryptonight::SplitContext* contexts,
                               const CnGeometry& geometry,
                               const cudaDeviceProp& props)
{
    if (count == 0U) return;
    const bool use_ttable = geometry.aes_backend == 1;
    if (use_ttable) {
        launch_cn_setup_selected<VariantIndex>(
            stream, device_id, count, states, scratchpads, contexts);
        return;
    }
    const int threads = cn_setup_occupancy_threads<VariantIndex, false>(props);
    const std::size_t total_threads = count * 8U;
    const int blocks =
        static_cast<int>((total_threads + threads - 1U) / threads);
    cryptonight_setup_stage_cooperative8<VariantIndex, false>
        <<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(),
               "GhostRider CUDA staggered CN setup launch failed");
}

template <std::uint8_t VariantIndex>
void launch_cn_loop_partition(cudaStream_t stream,
                              int device_id,
                              std::size_t count,
                              std::uint8_t* states,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              const CnGeometry& geometry,
                              int threads_override)
{
    (void)states;
    (void)geometry;
    if (count == 0U) return;

    int mode = g_cn_hardened_mode[device_id][VariantIndex];
    int threads = threads_override;
    if (mode == 0) mode = 4;
    if (threads <= 0)
        threads = g_cn_hardened_threads[device_id][VariantIndex];
    if (threads <= 0) threads = 128;

    cn_launch_mode<VariantIndex>(
        stream, mode, threads, count, scratchpads, contexts);
}

template <std::uint8_t VariantIndex>
void launch_cn_final_partition(cudaStream_t stream,
                               int device_id,
                               std::size_t count,
                               std::uint8_t* states,
                               std::uint8_t* scratchpads,
                               cryptonight::SplitContext* contexts,
                               const CnGeometry& geometry,
                               const cudaDeviceProp& props)
{
    if (count == 0U) return;
    const bool use_ttable = geometry.aes_backend == 1;
    if (use_ttable) {
        launch_cn_final_selected<VariantIndex>(
            stream, device_id, count, states, scratchpads, contexts);
        return;
    }
    const int threads = cn_final_occupancy_threads<VariantIndex, false>(props);
    const std::size_t total_threads = count * 8U;
    const int blocks =
        static_cast<int>((total_threads + threads - 1U) / threads);
    cryptonight_final_stage_cooperative8<VariantIndex, false>
        <<<blocks, threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(),
               "GhostRider CUDA staggered CN final launch failed");
}

template <std::uint8_t VariantIndex>
void launch_cn_staggered_mode(cudaStream_t primary,
                              int device_id,
                              int mode,
                              std::size_t count,
                              std::uint8_t* states,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              const CnGeometry& geometry,
                              const cudaDeviceProp& props,
                              int threads_a = 0,
                              int threads_b = 0)
{
    if (!cn_stagger_split_mode(mode)) {
        launch_cn_partition<VariantIndex>(
            primary, device_id, count, states, scratchpads, contexts,
            geometry, props);
        return;
    }

    const std::size_t first = cn_stagger_first_count(count, mode);
    if (first == 0U || first >= count) {
        launch_cn_partition<VariantIndex>(
            primary, device_id, count, states, scratchpads, contexts,
            geometry, props);
        return;
    }

    const std::size_t second = count - first;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    auto* states_b = states + first * kStateBytes;
    auto* scratch_b = scratchpads + first * cfg.page_size;
    auto* contexts_b = contexts + first;
    cudaStream_t secondary = cn_overlap_stream(device_id);
    const bool overlap_loops = cn_stagger_loop_overlap(mode);

    if (threads_a <= 0)
        threads_a = g_cn_hardened_threads[device_id][VariantIndex];
    if (threads_b <= 0)
        threads_b = g_cn_hardened_threads[device_id][VariantIndex];
    if (threads_a <= 0) threads_a = 128;
    if (threads_b <= 0) threads_b = 128;

    cudaEvent_t ready{}, done_b{};
    check_cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming),
               "cudaEventCreate stagger ready failed");
    check_cuda(cudaEventCreateWithFlags(&done_b, cudaEventDisableTiming),
               "cudaEventCreate stagger done failed");

    check_cuda(cudaEventRecord(ready, primary),
               "cudaEventRecord stagger ready failed");
    check_cuda(cudaStreamWaitEvent(secondary, ready, 0),
               "cudaStreamWaitEvent stagger ready failed");

    if (overlap_loops) {
        launch_cn_setup_partition<VariantIndex>(
            primary, device_id, first, states, scratchpads, contexts,
            geometry, props);
        launch_cn_setup_partition<VariantIndex>(
            secondary, device_id, second, states_b, scratch_b, contexts_b,
            geometry, props);

        launch_cn_loop_partition<VariantIndex>(
            primary, device_id, first, states, scratchpads, contexts,
            geometry, threads_a);
        launch_cn_loop_partition<VariantIndex>(
            secondary, device_id, second, states_b, scratch_b, contexts_b,
            geometry, threads_b);

        launch_cn_final_partition<VariantIndex>(
            primary, device_id, first, states, scratchpads, contexts,
            geometry, props);
        launch_cn_final_partition<VariantIndex>(
            secondary, device_id, second, states_b, scratch_b, contexts_b,
            geometry, props);
    } else {
        cudaEvent_t setup_a_done{}, loop_a_done{};
        check_cuda(
            cudaEventCreateWithFlags(&setup_a_done, cudaEventDisableTiming),
            "cudaEventCreate stagger setup failed");
        check_cuda(
            cudaEventCreateWithFlags(&loop_a_done, cudaEventDisableTiming),
            "cudaEventCreate stagger loop failed");

        launch_cn_setup_partition<VariantIndex>(
            primary, device_id, first, states, scratchpads, contexts,
            geometry, props);
        check_cuda(cudaEventRecord(setup_a_done, primary),
                   "cudaEventRecord stagger setup failed");
        check_cuda(cudaStreamWaitEvent(secondary, setup_a_done, 0),
                   "cudaStreamWaitEvent stagger setup failed");

        launch_cn_loop_partition<VariantIndex>(
            primary, device_id, first, states, scratchpads, contexts,
            geometry, threads_a);
        check_cuda(cudaEventRecord(loop_a_done, primary),
                   "cudaEventRecord stagger loop failed");

        launch_cn_setup_partition<VariantIndex>(
            secondary, device_id, second, states_b, scratch_b, contexts_b,
            geometry, props);
        check_cuda(cudaStreamWaitEvent(secondary, loop_a_done, 0),
                   "cudaStreamWaitEvent stagger serialize loops failed");
        launch_cn_loop_partition<VariantIndex>(
            secondary, device_id, second, states_b, scratch_b, contexts_b,
            geometry, threads_b);

        launch_cn_final_partition<VariantIndex>(
            primary, device_id, first, states, scratchpads, contexts,
            geometry, props);
        launch_cn_final_partition<VariantIndex>(
            secondary, device_id, second, states_b, scratch_b, contexts_b,
            geometry, props);

        cudaEventDestroy(loop_a_done);
        cudaEventDestroy(setup_a_done);
    }

    check_cuda(cudaEventRecord(done_b, secondary),
               "cudaEventRecord stagger done failed");
    check_cuda(cudaStreamWaitEvent(primary, done_b, 0),
               "cudaStreamWaitEvent stagger done failed");

    cudaEventDestroy(done_b);
    cudaEventDestroy(ready);
}

template <std::uint8_t VariantIndex>
std::filesystem::path cn_stagger_cache_path(
    int device_id,
    const cudaDeviceProp& props,
    std::size_t count)
{
    int driver = 0, runtime = 0;
    if (cudaDriverGetVersion(&driver) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key =
        cn_cache_key(device_id, props, count, driver, runtime);
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    return cn_cache_directory() /
        ("cn-stagger-rev" + std::to_string(kCnStaggerSchedulerRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) +
         "-p" + std::to_string(cfg.page_size) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_stagger_cache(
    int device_id,
    const cudaDeviceProp& props,
    std::size_t count)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;

    const auto path =
        cn_stagger_cache_path<VariantIndex>(device_id, props, count);
    if (path.empty()) return false;

    std::ifstream in(path);
    std::string magic;
    int revision = 0, mode = 0, threads_a = 0, threads_b = 0;
    if (!(in >> magic >> revision >> mode >> threads_a >> threads_b) ||
        magic != "YERBAS_CN_STAGGER" ||
        revision != kCnStaggerSchedulerRevision ||
        (mode != 1 && !cn_stagger_split_mode(mode)))
        return false;

    if (mode != 1) {
        int production_mode =
            g_cn_hardened_mode[device_id][VariantIndex];
        if (production_mode == 0) production_mode = 4;
        if (!cn_geometry_threads_valid_variant<VariantIndex>(
                production_mode, threads_a, props) ||
            !cn_geometry_threads_valid_variant<VariantIndex>(
                production_mode, threads_b, props))
            return false;
    }

    g_cn_stagger_mode[device_id][VariantIndex] = mode;
    g_cn_stagger_threads_a[device_id][VariantIndex] = threads_a;
    g_cn_stagger_threads_b[device_id][VariantIndex] = threads_b;
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_stagger_cache(
    int device_id,
    const cudaDeviceProp& props,
    std::size_t count)
{
    try {
        const auto path =
            cn_stagger_cache_path<VariantIndex>(device_id, props, count);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (out) {
            out << "YERBAS_CN_STAGGER "
                << kCnStaggerSchedulerRevision << ' '
                << g_cn_stagger_mode[device_id][VariantIndex] << ' '
                << g_cn_stagger_threads_a[device_id][VariantIndex] << ' '
                << g_cn_stagger_threads_b[device_id][VariantIndex] << '\n';
        }
    } catch (...) {}
}

template <std::uint8_t VariantIndex>
float time_cn_staggered_mode(
    cudaStream_t stream,
    int device_id,
    int mode,
    std::size_t count,
    std::uint8_t* states,
    std::uint8_t* scratchpads,
    cryptonight::SplitContext* contexts,
    const CnGeometry& geometry,
    const cudaDeviceProp& props,
    int threads_a = 0,
    int threads_b = 0)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault),
               "cudaEventCreate stagger timing start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault),
               "cudaEventCreate stagger timing stop failed");
    check_cuda(cudaEventRecord(start, stream),
               "cudaEventRecord stagger timing start failed");

    launch_cn_staggered_mode<VariantIndex>(
        stream, device_id, mode, count, states, scratchpads, contexts,
        geometry, props, threads_a, threads_b);

    check_cuda(cudaEventRecord(stop, stream),
               "cudaEventRecord stagger timing stop failed");
    check_cuda(cudaEventSynchronize(stop),
               "cudaEventSynchronize stagger timing failed");

    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop),
               "cudaEventElapsedTime stagger timing failed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
int build_cn_stagger_geometry_candidates(
    int mode,
    int baseline_threads,
    const cudaDeviceProp& props,
    std::array<int, kCnStaggerGeometryMaxCandidates>& candidates)
{
    int count = 0;
    const int warp = std::max(32, props.warpSize);
    const int max_threads =
        std::max(warp, (props.maxThreadsPerBlock / warp) * warp);
    const auto align_warp = [warp](int threads) {
        return std::max(warp, (threads / warp) * warp);
    };
    const auto add = [&](int threads) {
        threads = align_warp(threads);
        if (!cn_geometry_threads_valid_variant<VariantIndex>(
                mode, threads, props))
            return;
        for (int i = 0; i < count; ++i)
            if (candidates[i] == threads) return;
        if (count < kCnStaggerGeometryMaxCandidates)
            candidates[count++] = threads;
    };

    add(baseline_threads);
    add(baseline_threads - warp);
    add(baseline_threads + warp);
    add((max_threads * 5) / 8);
    add((max_threads * 3) / 4);
    add((max_threads * 7) / 8);
    add(max_threads);
    return count;
}

template <std::uint8_t VariantIndex, typename Measure>
void tune_cn_stagger_partition_geometry(
    int device_id,
    const cudaDeviceProp& props,
    int mode,
    Measure&& measure)
{
    int production_mode =
        g_cn_hardened_mode[device_id][VariantIndex];
    if (production_mode == 0) production_mode = 4;

    int baseline_threads =
        g_cn_hardened_threads[device_id][VariantIndex];
    if (baseline_threads <= 0) baseline_threads = 128;

    std::array<int, kCnStaggerGeometryMaxCandidates> candidates{};
    const int candidate_count =
        build_cn_stagger_geometry_candidates<VariantIndex>(
            production_mode, baseline_threads, props, candidates);

    if (candidate_count <= 0) {
        g_cn_stagger_threads_a[device_id][VariantIndex] =
            baseline_threads;
        g_cn_stagger_threads_b[device_id][VariantIndex] =
            baseline_threads;
        return;
    }

    int best_a = baseline_threads;
    float best_a_ms = 1.0e30F;
    for (int i = 0; i < candidate_count; ++i) {
        const int threads = candidates[i];
        (void)measure(mode, threads, baseline_threads);
        std::array<float, kCnStaggerGeometryPasses> samples{};
        for (int pass = 0; pass < kCnStaggerGeometryPasses; ++pass)
            samples[pass] =
                measure(mode, threads, baseline_threads);
        const float median = cn_stagger_median(samples);
        if (median < best_a_ms) {
            best_a_ms = median;
            best_a = threads;
        }
    }

    int best_b = baseline_threads;
    float best_b_ms = 1.0e30F;
    for (int i = 0; i < candidate_count; ++i) {
        const int threads = candidates[i];
        (void)measure(mode, best_a, threads);
        std::array<float, kCnStaggerGeometryPasses> samples{};
        for (int pass = 0; pass < kCnStaggerGeometryPasses; ++pass)
            samples[pass] = measure(mode, best_a, threads);
        const float median = cn_stagger_median(samples);
        if (median < best_b_ms) {
            best_b_ms = median;
            best_b = threads;
        }
    }

    std::array<float, kCnStaggerGeometryConfirmPasses> baseline_samples{};
    std::array<float, kCnStaggerGeometryConfirmPasses> tuned_samples{};
    for (int pass = 0; pass < kCnStaggerGeometryConfirmPasses; ++pass) {
        if ((pass & 1) == 0) {
            baseline_samples[pass] =
                measure(mode, baseline_threads, baseline_threads);
            tuned_samples[pass] =
                measure(mode, best_a, best_b);
        } else {
            tuned_samples[pass] =
                measure(mode, best_a, best_b);
            baseline_samples[pass] =
                measure(mode, baseline_threads, baseline_threads);
        }
    }

    const float baseline_ms = cn_stagger_median(baseline_samples);
    const float tuned_ms = cn_stagger_median(tuned_samples);
    const bool promoted =
        tuned_ms < baseline_ms * kCnStaggerGeometryPromotionRatio;
    const int proposed_a = best_a;
    const int proposed_b = best_b;

    if (!promoted) {
        best_a = baseline_threads;
        best_b = baseline_threads;
    }

    g_cn_stagger_threads_a[device_id][VariantIndex] = best_a;
    g_cn_stagger_threads_b[device_id][VariantIndex] = best_b;

    const float gain = baseline_ms > 0.0F
        ? ((baseline_ms - tuned_ms) / baseline_ms) * 100.0F
        : 0.0F;

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN stagger geometry] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | mode=";
    if (cn_stagger_loop_overlap(mode))
        std::cout << "overlap" << cn_stagger_ratio(mode);
    else
        std::cout << "stagger" << cn_stagger_ratio(mode);
    std::cout << " | baseline=" << baseline_threads << '/'
              << baseline_threads
              << " | proposed=" << proposed_a << '/' << proposed_b
              << " | selected=" << best_a << '/' << best_b
              << " | baseline-ms=" << baseline_ms
              << " | tuned-ms=" << tuned_ms
              << " | gain=" << gain << "%"
              << " | threshold=1%"
              << " | promoted=" << (promoted ? "yes" : "no")
              << " | candidates=" << candidate_count
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex>
bool launch_cn_staggered_if_ready(
    cudaStream_t stream,
    int device_id,
    std::size_t count,
    std::uint8_t* states,
    std::uint8_t* scratchpads,
    cryptonight::SplitContext* contexts,
    const CnGeometry& geometry,
    const cudaDeviceProp& props)
{
    if (cn_stagger_disabled() || count < 96U) return false;
    const bool use_ttable = geometry.aes_backend == 1;
    if (!cn_overlap_selectors_ready<VariantIndex>(
            device_id, use_ttable))
        return false;

    if (device_id >= 0 &&
        device_id < kCnPhaseProfileMaxDevices &&
        cn_phase_diagnostics_enabled() &&
        g_cn_phase_samples[device_id][VariantIndex] == 0U) {
        launch_split_cryptonight_variant_phase_profiled_legacy<
            VariantIndex>(
                stream, states, count, scratchpads, contexts, geometry);
        return true;
    }

    int& mode = g_cn_stagger_mode[device_id][VariantIndex];
    if (mode == 0 &&
        !load_cn_stagger_cache<VariantIndex>(
            device_id, props, count)) {
        if (!cn_stagger_tuning_requested()) return false;

        std::uint8_t* saved_states = nullptr;
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&saved_states),
                count * kStateBytes),
            "cudaMalloc stagger saved states failed");
        check_cuda(
            cudaMemcpyAsync(
                saved_states, states, count * kStateBytes,
                cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpy stagger save states failed");
        check_cuda(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize stagger save failed");

        const auto reset = [&]() {
            check_cuda(
                cudaMemcpyAsync(
                    states, saved_states, count * kStateBytes,
                    cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpy stagger restore states failed");
            check_cuda(
                cudaStreamSynchronize(stream),
                "cudaStreamSynchronize stagger restore failed");
        };

        const auto measure = [&](int candidate_mode,
                                 int threads_a = 0,
                                 int threads_b = 0) {
            reset();
            return time_cn_staggered_mode<VariantIndex>(
                stream, device_id, candidate_mode, count,
                states, scratchpads, contexts, geometry, props,
                threads_a, threads_b);
        };

        for (const int candidate_mode : kCnStaggerModes)
            (void)measure(candidate_mode);

        std::array<
            std::array<float, kCnStaggerTunePasses>,
            kCnStaggerModes.size()> samples{};
        for (int pass = 0; pass < kCnStaggerTunePasses; ++pass) {
            for (std::size_t position = 0;
                 position < kCnStaggerModes.size();
                 ++position) {
                const std::size_t candidate_index =
                    (position + static_cast<std::size_t>(pass)) %
                    kCnStaggerModes.size();
                samples[candidate_index]
                       [static_cast<std::size_t>(pass)] =
                    measure(kCnStaggerModes[candidate_index]);
            }
        }

        std::array<float, kCnStaggerModes.size()> medians{};
        for (std::size_t i = 0;
             i < kCnStaggerModes.size();
             ++i)
            medians[i] = cn_stagger_median(samples[i]);

        const float single_ms = medians[0];
        mode = 1;
        float best_ms = single_ms;
        for (std::size_t i = 1;
             i < kCnStaggerModes.size();
             ++i) {
            if (medians[i] <
                    single_ms * kCnStaggerPromotionRatio &&
                medians[i] < best_ms) {
                mode = kCnStaggerModes[i];
                best_ms = medians[i];
            }
        }

        g_cn_stagger_threads_a[device_id][VariantIndex] = 0;
        g_cn_stagger_threads_b[device_id][VariantIndex] = 0;

        if (mode != 1) {
            tune_cn_stagger_partition_geometry<VariantIndex>(
                device_id, props, mode, measure);
        }

        reset();
        cudaFree(saved_states);
        save_cn_stagger_cache<VariantIndex>(
            device_id, props, count);

        const float gain = single_ms > 0.0F
            ? ((single_ms - best_ms) / single_ms) * 100.0F
            : 0.0F;

        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA CN stagger tuner] GPU " << device_id
                  << " | "
                  << cryptonight::config_value(VariantIndex).name
                  << " | batch=" << count
                  << " | single=" << medians[0] << " ms"
                  << " | stagger25=" << medians[1] << " ms"
                  << " | stagger33=" << medians[2] << " ms"
                  << " | stagger40=" << medians[3] << " ms"
                  << " | stagger50=" << medians[4] << " ms"
                  << " | stagger60=" << medians[5] << " ms"
                  << " | stagger67=" << medians[6] << " ms"
                  << " | stagger75=" << medians[7] << " ms"
                  << " | overlap25=" << medians[8] << " ms"
                  << " | overlap50=" << medians[9] << " ms"
                  << " | overlap75=" << medians[10] << " ms"
                  << " | passes=" << kCnStaggerTunePasses
                  << " | warmup=1/candidate"
                  << " | order=fully-rotated"
                  << " | threshold=3%"
                  << " | gain=" << gain << "%"
                  << " | production=";

        if (mode == 1) {
            std::cout << "single";
        } else if (cn_stagger_loop_overlap(mode)) {
            std::cout << "overlap"
                      << cn_stagger_ratio(mode);
        } else {
            std::cout << "stagger"
                      << cn_stagger_ratio(mode);
        }

        if (mode != 1) {
            std::cout << " | threads="
                      << g_cn_stagger_threads_a
                             [device_id][VariantIndex]
                      << '/'
                      << g_cn_stagger_threads_b
                             [device_id][VariantIndex];
        }
        std::cout << std::defaultfloat << '\n';
    }

    if (mode == 1) return false;

    const int threads_a =
        g_cn_stagger_threads_a[device_id][VariantIndex];
    const int threads_b =
        g_cn_stagger_threads_b[device_id][VariantIndex];

    launch_cn_staggered_mode<VariantIndex>(
        stream, device_id, mode, count, states, scratchpads, contexts,
        geometry, props, threads_a, threads_b);

    if (!g_cn_stagger_reported[device_id][VariantIndex]) {
        g_cn_stagger_reported[device_id][VariantIndex] = true;
        const int ratio = cn_stagger_ratio(mode);
        const std::size_t first =
            cn_stagger_first_count(count, mode);
        std::cout << "[CUDA CN stagger] GPU " << device_id
                  << " | "
                  << cryptonight::config_value(VariantIndex).name
                  << " | mode=staggered"
                  << " | ratio=" << ratio << '/'
                  << (100 - ratio)
                  << " | batch=" << count
                  << " | partitions=" << first << '+'
                  << (count - first)
                  << " | threads=" << threads_a << '/'
                  << threads_b
                  << " | source=cache-or-tuned"
                  << " | loop-overlap="
                  << (cn_stagger_loop_overlap(mode)
                          ? "yes"
                          : "no")
                  << " | generic=yes\n";
    }
    return true;
}

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant_staggered(
    cudaStream_t stream,
    std::uint8_t* states,
    std::size_t count,
    std::uint8_t* scratchpads,
    cryptonight::SplitContext* contexts,
    const CnGeometry& geometry)
{
    int device_id = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDevice(&device_id),
               "cudaGetDevice CN stagger failed");
    check_cuda(
        cudaGetDeviceProperties(&props, device_id),
        "cudaGetDeviceProperties CN stagger failed");
    if (launch_cn_staggered_if_ready<VariantIndex>(
            stream, device_id, count, states, scratchpads, contexts,
            geometry, props))
        return;

    launch_split_cryptonight_variant_phase_profiled_legacy<
        VariantIndex>(
            stream, states, count, scratchpads, contexts, geometry);
}

void launch_split_cryptonight_phase_profiled(
    cudaStream_t stream,
    std::uint8_t* states,
    std::size_t count,
    std::uint8_t variant,
    std::uint8_t* scratchpads,
    cryptonight::SplitContext* contexts,
    const CnGeometry& geometry)
{
    switch (variant) {
    case 0:
        launch_split_cryptonight_variant_staggered<0>(
            stream, states, count, scratchpads, contexts, geometry);
        break;
    case 1:
        launch_split_cryptonight_variant_staggered<1>(
            stream, states, count, scratchpads, contexts, geometry);
        break;
    case 2:
        launch_split_cryptonight_variant_staggered<2>(
            stream, states, count, scratchpads, contexts, geometry);
        break;
    case 3:
        launch_split_cryptonight_variant_staggered<3>(
            stream, states, count, scratchpads, contexts, geometry);
        break;
    case 4:
        launch_split_cryptonight_variant_staggered<4>(
            stream, states, count, scratchpads, contexts, geometry);
        break;
    case 5:
        launch_split_cryptonight_variant_staggered<5>(
            stream, states, count, scratchpads, contexts, geometry);
        break;
    default:
        throw std::runtime_error(
            "GhostRider CryptoNight stage has invalid variant index");
    }
}

} // anonymous namespace
