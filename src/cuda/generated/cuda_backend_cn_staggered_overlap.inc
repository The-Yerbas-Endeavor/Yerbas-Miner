// Generic staggered CryptoNight overlap scheduler.
//
// The established stagger path keeps the two memory-bound phase-2 loops
// serialized while hiding setup/final work around them. Rev6 also benchmarks a
// small set of concurrent-loop candidates. Those candidates run two independent
// CN partitions on separate CUDA streams so the hardware scheduler can hide a
// dependent scratchpad/multiply stall with work from another hash chain. Runtime
// timing decides whether that helps; no GPU model or architecture rules are used.
//
// Production policy is cache-first. With YERBAS_CUDA_OVERLAP unset or "auto",
// a previously proven stagger cache is used automatically, but a missing cache
// never pauses mining to benchmark. YERBAS_CUDA_OVERLAP=1/on/true/tune/full
// explicitly allows the runtime benchmark. 0/off/false/no disables stagger use.
//
// Tuning is intentionally order-balanced. Every candidate receives a discarded
// warm-up, then one measured pass per candidate position rotates execution order
// so each mode occupies every ordinal position exactly once. This prevents GPU
// boost/JIT/stream warm-up from making a fixed position look artificially faster.
//
// Serialized modes are encoded as their partition-A percentage (25..75).
// Concurrent-loop modes use 100+ratio (125/150/175). The encoding is internal;
// cache/reporting always exposes the actual split and loop-overlap policy.

namespace {

constexpr float kCnStaggerPromotionRatio = 0.97F;
constexpr int kCnStaggerSchedulerRevision = 6;
constexpr std::array<int, 11> kCnStaggerModes{{1, 25, 33, 40, 50, 60, 67, 75, 125, 150, 175}};
constexpr int kCnStaggerTunePasses = static_cast<int>(kCnStaggerModes.size());
static std::array<std::array<int, 6>, kCnPhaseProfileMaxDevices> g_cn_stagger_mode{};
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
    std::size_t first = ((count * static_cast<std::size_t>(ratio) / 100U) / 32U) * 32U;
    if (first == 0U) first = std::min<std::size_t>(32U, count);
    if (first >= count && count > 32U) first = count - 32U;
    return first;
}

inline float cn_stagger_median(std::array<float, kCnStaggerTunePasses> values)
{
    // Tiny fixed-size insertion sort avoids depending on candidate execution order.
    for (std::size_t i = 1; i < values.size(); ++i) {
        const float value = values[i];
        std::size_t j = i;
        while (j > 0U && values[j - 1U] > value) {
            values[j] = values[j - 1U];
            --j;
        }
        values[j] = value;
    }
    return values[values.size() / 2U];
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
        launch_cn_setup_selected<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
        return;
    }
    const int threads = cn_setup_occupancy_threads<VariantIndex, false>(props);
    const std::size_t total_threads = count * 8U;
    const int blocks = static_cast<int>((total_threads + threads - 1U) / threads);
    cryptonight_setup_stage_cooperative8<VariantIndex, false><<<blocks, threads, 0, stream>>>(
        states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA staggered CN setup launch failed");
}

template <std::uint8_t VariantIndex>
void launch_cn_loop_partition(cudaStream_t stream,
                              std::size_t count,
                              std::uint8_t* states,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              const CnGeometry& geometry)
{
    if (count == 0U) return;
    if (geometry.aes_backend == 1)
        launch_cn_loop_block_tuned<VariantIndex, true>(stream, states, count, scratchpads, contexts, geometry);
    else
        launch_cn_loop_block_tuned<VariantIndex, false>(stream, states, count, scratchpads, contexts, geometry);
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
        launch_cn_final_selected<VariantIndex>(stream, device_id, count, states, scratchpads, contexts);
        return;
    }
    const int threads = cn_final_occupancy_threads<VariantIndex, false>(props);
    const std::size_t total_threads = count * 8U;
    const int blocks = static_cast<int>((total_threads + threads - 1U) / threads);
    cryptonight_final_stage_cooperative8<VariantIndex, false><<<blocks, threads, 0, stream>>>(
        states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA staggered CN final launch failed");
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
                              const cudaDeviceProp& props)
{
    if (!cn_stagger_split_mode(mode)) {
        launch_cn_partition<VariantIndex>(primary, device_id, count,
                                          states, scratchpads, contexts, geometry, props);
        return;
    }

    const std::size_t first = cn_stagger_first_count(count, mode);
    if (first == 0U || first >= count) {
        launch_cn_partition<VariantIndex>(primary, device_id, count,
                                          states, scratchpads, contexts, geometry, props);
        return;
    }

    const std::size_t second = count - first;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    auto* states_b = states + first * kStateBytes;
    auto* scratch_b = scratchpads + first * cfg.page_size;
    auto* contexts_b = contexts + first;
    cudaStream_t secondary = cn_overlap_stream(device_id);
    const bool overlap_loops = cn_stagger_loop_overlap(mode);

    cudaEvent_t ready{}, done_b{};
    check_cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cudaEventCreate stagger ready failed");
    check_cuda(cudaEventCreateWithFlags(&done_b, cudaEventDisableTiming), "cudaEventCreate stagger done failed");

    check_cuda(cudaEventRecord(ready, primary), "cudaEventRecord stagger ready failed");
    check_cuda(cudaStreamWaitEvent(secondary, ready, 0), "cudaStreamWaitEvent stagger ready failed");

    if (overlap_loops) {
        // Each stream owns a disjoint state/scratch/context range. Preserve the
        // exact setup->loop->final ordering within each partition but remove the
        // cross-stream phase-2 dependency. CUDA may then co-reside/interleave
        // the two independent loop kernels when the current hardware benefits.
        launch_cn_setup_partition<VariantIndex>(primary, device_id, first,
                                                states, scratchpads, contexts, geometry, props);
        launch_cn_setup_partition<VariantIndex>(secondary, device_id, second,
                                                states_b, scratch_b, contexts_b, geometry, props);

        launch_cn_loop_partition<VariantIndex>(primary, first,
                                               states, scratchpads, contexts, geometry);
        launch_cn_loop_partition<VariantIndex>(secondary, second,
                                               states_b, scratch_b, contexts_b, geometry);

        launch_cn_final_partition<VariantIndex>(primary, device_id, first,
                                                states, scratchpads, contexts, geometry, props);
        launch_cn_final_partition<VariantIndex>(secondary, device_id, second,
                                                states_b, scratch_b, contexts_b, geometry, props);
    } else {
        // Established stagger schedule: overlap setup/final work, but serialize
        // the two phase-2 loops so they never contend with one another.
        cudaEvent_t setup_a_done{}, loop_a_done{};
        check_cuda(cudaEventCreateWithFlags(&setup_a_done, cudaEventDisableTiming),
                   "cudaEventCreate stagger setup failed");
        check_cuda(cudaEventCreateWithFlags(&loop_a_done, cudaEventDisableTiming),
                   "cudaEventCreate stagger loop failed");

        launch_cn_setup_partition<VariantIndex>(primary, device_id, first,
                                                states, scratchpads, contexts, geometry, props);
        check_cuda(cudaEventRecord(setup_a_done, primary), "cudaEventRecord stagger setup failed");
        check_cuda(cudaStreamWaitEvent(secondary, setup_a_done, 0),
                   "cudaStreamWaitEvent stagger setup failed");

        launch_cn_loop_partition<VariantIndex>(primary, first,
                                               states, scratchpads, contexts, geometry);
        check_cuda(cudaEventRecord(loop_a_done, primary), "cudaEventRecord stagger loop failed");

        launch_cn_setup_partition<VariantIndex>(secondary, device_id, second,
                                                states_b, scratch_b, contexts_b, geometry, props);
        check_cuda(cudaStreamWaitEvent(secondary, loop_a_done, 0),
                   "cudaStreamWaitEvent stagger serialize loops failed");
        launch_cn_loop_partition<VariantIndex>(secondary, second,
                                               states_b, scratch_b, contexts_b, geometry);

        launch_cn_final_partition<VariantIndex>(primary, device_id, first,
                                                states, scratchpads, contexts, geometry, props);
        launch_cn_final_partition<VariantIndex>(secondary, device_id, second,
                                                states_b, scratch_b, contexts_b, geometry, props);

        cudaEventDestroy(loop_a_done);
        cudaEventDestroy(setup_a_done);
    }

    check_cuda(cudaEventRecord(done_b, secondary), "cudaEventRecord stagger done failed");
    check_cuda(cudaStreamWaitEvent(primary, done_b, 0), "cudaStreamWaitEvent stagger done failed");

    cudaEventDestroy(done_b);
    cudaEventDestroy(ready);
}

template <std::uint8_t VariantIndex>
std::filesystem::path cn_stagger_cache_path(int device_id,
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
        ("cn-stagger-rev" + std::to_string(kCnStaggerSchedulerRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) +
         "-p" + std::to_string(cfg.page_size) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_stagger_cache(int device_id, const cudaDeviceProp& props, std::size_t count)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_stagger_cache_path<VariantIndex>(device_id, props, count);
    if (path.empty()) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0, mode = 0;
    if (!(in >> magic >> revision >> mode) || magic != "YERBAS_CN_STAGGER" ||
        revision != kCnStaggerSchedulerRevision ||
        (mode != 1 && !cn_stagger_split_mode(mode))) return false;
    g_cn_stagger_mode[device_id][VariantIndex] = mode;
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_stagger_cache(int device_id, const cudaDeviceProp& props, std::size_t count)
{
    try {
        const auto path = cn_stagger_cache_path<VariantIndex>(device_id, props, count);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (out) out << "YERBAS_CN_STAGGER " << kCnStaggerSchedulerRevision << ' '
                     << g_cn_stagger_mode[device_id][VariantIndex] << '\n';
    } catch (...) {}
}

template <std::uint8_t VariantIndex>
float time_cn_staggered_mode(cudaStream_t stream,
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
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate stagger timing start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate stagger timing stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord stagger timing start failed");
    launch_cn_staggered_mode<VariantIndex>(stream, device_id, mode, count,
                                           states, scratchpads, contexts, geometry, props);
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stagger timing stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stagger timing failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime stagger timing failed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
bool launch_cn_staggered_if_ready(cudaStream_t stream,
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
    if (!cn_overlap_selectors_ready<VariantIndex>(device_id, use_ttable)) return false;

    if (device_id >= 0 && device_id < kCnPhaseProfileMaxDevices &&
        cn_phase_diagnostics_enabled() &&
        g_cn_phase_samples[device_id][VariantIndex] == 0U) {
        launch_split_cryptonight_variant_phase_profiled_legacy<VariantIndex>(
            stream, states, count, scratchpads, contexts, geometry);
        return true;
    }

    int& mode = g_cn_stagger_mode[device_id][VariantIndex];
    if (mode == 0 && !load_cn_stagger_cache<VariantIndex>(device_id, props, count)) {
        // Normal production is cache-only. Explicit overlap tuning is the only
        // path allowed to benchmark candidates and create a new cache.
        if (!cn_stagger_tuning_requested()) return false;

        std::uint8_t* saved_states = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&saved_states), count * kStateBytes),
                   "cudaMalloc stagger saved states failed");
        check_cuda(cudaMemcpyAsync(saved_states, states, count * kStateBytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy stagger save states failed");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize stagger save failed");

        const auto reset = [&]() {
            check_cuda(cudaMemcpyAsync(states, saved_states, count * kStateBytes,
                                       cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpy stagger restore states failed");
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize stagger restore failed");
        };
        const auto measure = [&](int candidate_mode) {
            reset();
            return time_cn_staggered_mode<VariantIndex>(
                stream, device_id, candidate_mode, count,
                states, scratchpads, contexts, geometry, props);
        };

        for (const int candidate_mode : kCnStaggerModes)
            (void)measure(candidate_mode);

        std::array<std::array<float, kCnStaggerTunePasses>, kCnStaggerModes.size()> samples{};
        for (int pass = 0; pass < kCnStaggerTunePasses; ++pass) {
            for (std::size_t position = 0; position < kCnStaggerModes.size(); ++position) {
                const std::size_t candidate_index =
                    (position + static_cast<std::size_t>(pass)) % kCnStaggerModes.size();
                samples[candidate_index][static_cast<std::size_t>(pass)] =
                    measure(kCnStaggerModes[candidate_index]);
            }
        }

        std::array<float, kCnStaggerModes.size()> medians{};
        for (std::size_t i = 0; i < kCnStaggerModes.size(); ++i)
            medians[i] = cn_stagger_median(samples[i]);

        const float single_ms = medians[0];
        mode = 1;
        float best_ms = single_ms;
        for (std::size_t i = 1; i < kCnStaggerModes.size(); ++i) {
            if (medians[i] < single_ms * kCnStaggerPromotionRatio && medians[i] < best_ms) {
                mode = kCnStaggerModes[i];
                best_ms = medians[i];
            }
        }

        reset();
        cudaFree(saved_states);
        save_cn_stagger_cache<VariantIndex>(device_id, props, count);

        const float gain = single_ms > 0.0F ? ((single_ms - best_ms) / single_ms) * 100.0F : 0.0F;
        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA CN stagger tuner] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
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
                  << " | order=rotated"
                  << " | threshold=3%"
                  << " | gain=" << gain << "%"
                  << " | production=";
        if (mode == 1) {
            std::cout << "single";
        } else if (cn_stagger_loop_overlap(mode)) {
            std::cout << "overlap" << cn_stagger_ratio(mode);
        } else {
            std::cout << "stagger" << cn_stagger_ratio(mode);
        }
        std::cout << std::defaultfloat << '\n';
    }

    if (mode == 1) return false;

    launch_cn_staggered_mode<VariantIndex>(stream, device_id, mode, count,
                                           states, scratchpads, contexts, geometry, props);
    if (!g_cn_stagger_reported[device_id][VariantIndex]) {
        g_cn_stagger_reported[device_id][VariantIndex] = true;
        const int ratio = cn_stagger_ratio(mode);
        const std::size_t first = cn_stagger_first_count(count, mode);
        std::cout << "[CUDA CN stagger] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | mode=staggered"
                  << " | ratio=" << ratio << '/' << (100 - ratio)
                  << " | batch=" << count
                  << " | partitions=" << first << '+' << (count - first)
                  << " | source=cache-or-tuned"
                  << " | loop-overlap=" << (cn_stagger_loop_overlap(mode) ? "yes" : "no")
                  << " | generic=yes\n";
    }
    return true;
}

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant_staggered(cudaStream_t stream,
                                                std::uint8_t* states,
                                                std::size_t count,
                                                std::uint8_t* scratchpads,
                                                cryptonight::SplitContext* contexts,
                                                const CnGeometry& geometry)
{
    int device_id = 0;
    cudaDeviceProp props{};
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN stagger failed");
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN stagger failed");
    if (launch_cn_staggered_if_ready<VariantIndex>(stream, device_id, count,
                                                   states, scratchpads, contexts,
                                                   geometry, props)) return;
    launch_split_cryptonight_variant_phase_profiled_legacy<VariantIndex>(
        stream, states, count, scratchpads, contexts, geometry);
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
    case 0: launch_split_cryptonight_variant_staggered<0>(stream,states,count,scratchpads,contexts,geometry); break;
    case 1: launch_split_cryptonight_variant_staggered<1>(stream,states,count,scratchpads,contexts,geometry); break;
    case 2: launch_split_cryptonight_variant_staggered<2>(stream,states,count,scratchpads,contexts,geometry); break;
    case 3: launch_split_cryptonight_variant_staggered<3>(stream,states,count,scratchpads,contexts,geometry); break;
    case 4: launch_split_cryptonight_variant_staggered<4>(stream,states,count,scratchpads,contexts,geometry); break;
    case 5: launch_split_cryptonight_variant_staggered<5>(stream,states,count,scratchpads,contexts,geometry); break;
    default: throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
    }
}

} // anonymous namespace
