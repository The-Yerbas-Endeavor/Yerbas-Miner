// Generic staggered CryptoNight overlap scheduler.
//
// This scheduler deliberately avoids overlapping the two memory-bound phase-2
// loops. Partition B setup overlaps partition A loop; partition A final overlaps
// partition B loop. Runtime timing decides whether 33/67 or 50/50 staggered
// execution beats the normal single-stream path. No GPU model rules are used.

namespace {

constexpr float kCnStaggerPromotionRatio = 0.97F;
constexpr int kCnStaggerSchedulerRevision = 1;
static std::array<std::array<int, 6>, kCnPhaseProfileMaxDevices> g_cn_stagger_mode{};
static std::array<std::array<bool, 6>, kCnPhaseProfileMaxDevices> g_cn_stagger_reported{};

inline float cn_stagger_median3(float a, float b, float c)
{
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return b;
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
    auto* states_b = states + first * kStateBytes;
    auto* scratch_b = scratchpads + first * cfg.page_size;
    auto* contexts_b = contexts + first;
    cudaStream_t secondary = cn_overlap_stream(device_id);

    cudaEvent_t ready{}, setup_a_done{}, loop_a_done{}, done_b{};
    check_cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cudaEventCreate stagger ready failed");
    check_cuda(cudaEventCreateWithFlags(&setup_a_done, cudaEventDisableTiming), "cudaEventCreate stagger setup failed");
    check_cuda(cudaEventCreateWithFlags(&loop_a_done, cudaEventDisableTiming), "cudaEventCreate stagger loop failed");
    check_cuda(cudaEventCreateWithFlags(&done_b, cudaEventDisableTiming), "cudaEventCreate stagger done failed");

    // Preserve ordering with work already queued on the caller's stream.
    check_cuda(cudaEventRecord(ready, primary), "cudaEventRecord stagger ready failed");
    check_cuda(cudaStreamWaitEvent(secondary, ready, 0), "cudaStreamWaitEvent stagger ready failed");

    // A.setup first. Once it is complete, B.setup may overlap A.loop.
    launch_cn_setup_partition<VariantIndex>(primary, device_id, first,
                                            states, scratchpads, contexts, geometry, props);
    check_cuda(cudaEventRecord(setup_a_done, primary), "cudaEventRecord stagger setup failed");
    check_cuda(cudaStreamWaitEvent(secondary, setup_a_done, 0), "cudaStreamWaitEvent stagger setup failed");

    launch_cn_loop_partition<VariantIndex>(primary, first,
                                           states, scratchpads, contexts, geometry);
    check_cuda(cudaEventRecord(loop_a_done, primary), "cudaEventRecord stagger loop failed");

    launch_cn_setup_partition<VariantIndex>(secondary, device_id, second,
                                            states_b, scratch_b, contexts_b, geometry, props);
    // B.loop cannot start until A.loop is complete. Its own stream ordering also
    // guarantees B.setup has completed first.
    check_cuda(cudaStreamWaitEvent(secondary, loop_a_done, 0), "cudaStreamWaitEvent stagger serialize loops failed");
    launch_cn_loop_partition<VariantIndex>(secondary, second,
                                           states_b, scratch_b, contexts_b, geometry);

    // A.final follows A.loop and can execute while B.loop owns the memory path.
    launch_cn_final_partition<VariantIndex>(primary, device_id, first,
                                            states, scratchpads, contexts, geometry, props);
    launch_cn_final_partition<VariantIndex>(secondary, device_id, second,
                                            states_b, scratch_b, contexts_b, geometry, props);

    check_cuda(cudaEventRecord(done_b, secondary), "cudaEventRecord stagger done failed");
    check_cuda(cudaStreamWaitEvent(primary, done_b, 0), "cudaStreamWaitEvent stagger done failed");

    cudaEventDestroy(done_b);
    cudaEventDestroy(loop_a_done);
    cudaEventDestroy(setup_a_done);
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
        revision != kCnStaggerSchedulerRevision || (mode != 1 && mode != 33 && mode != 50)) return false;
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
    if (!cn_overlap_enabled() || count < 96U) return false;
    const bool use_ttable = geometry.aes_backend == 1;
    if (!cn_overlap_selectors_ready<VariantIndex>(device_id, use_ttable)) return false;

    int& mode = g_cn_stagger_mode[device_id][VariantIndex];
    if (mode == 0 && !load_cn_stagger_cache<VariantIndex>(device_id, props, count)) {
        std::uint8_t* saved_states = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&saved_states), count * kStateBytes),
                   "cudaMalloc stagger saved states failed");
        check_cuda(cudaMemcpyAsync(saved_states, states, count * kStateBytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy stagger save states failed");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize stagger save failed");

        auto reset = [&]() {
            check_cuda(cudaMemcpyAsync(states, saved_states, count * kStateBytes,
                                       cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpy stagger restore states failed");
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize stagger restore failed");
        };

        std::array<float, 3> single{}, stagger33{}, stagger50{};
        for (int pass = 0; pass < 3; ++pass) {
            reset();
            single[pass] = time_cn_staggered_mode<VariantIndex>(stream, device_id, 1, count,
                states, scratchpads, contexts, geometry, props);
            reset();
            stagger33[pass] = time_cn_staggered_mode<VariantIndex>(stream, device_id, 33, count,
                states, scratchpads, contexts, geometry, props);
            reset();
            stagger50[pass] = time_cn_staggered_mode<VariantIndex>(stream, device_id, 50, count,
                states, scratchpads, contexts, geometry, props);
        }

        const float single_ms = cn_stagger_median3(single[0], single[1], single[2]);
        const float split33_ms = cn_stagger_median3(stagger33[0], stagger33[1], stagger33[2]);
        const float split50_ms = cn_stagger_median3(stagger50[0], stagger50[1], stagger50[2]);
        mode = 1;
        float best_ms = single_ms;
        if (split33_ms < single_ms * kCnStaggerPromotionRatio && split33_ms < best_ms) {
            mode = 33;
            best_ms = split33_ms;
        }
        if (split50_ms < single_ms * kCnStaggerPromotionRatio && split50_ms < best_ms) {
            mode = 50;
            best_ms = split50_ms;
        }

        reset();
        cudaFree(saved_states);
        save_cn_stagger_cache<VariantIndex>(device_id, props, count);

        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA CN stagger tuner] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | batch=" << count
                  << " | single=" << single_ms << " ms"
                  << " | stagger33=" << split33_ms << " ms"
                  << " | stagger50=" << split50_ms << " ms"
                  << " | threshold=3%"
                  << " | production=" << (mode == 1 ? "single" : (mode == 33 ? "stagger33" : "stagger50"))
                  << std::defaultfloat << '\n';
    }

    launch_cn_staggered_mode<VariantIndex>(stream, device_id, mode, count,
                                           states, scratchpads, contexts, geometry, props);
    if (!g_cn_stagger_reported[device_id][VariantIndex]) {
        g_cn_stagger_reported[device_id][VariantIndex] = true;
        const std::size_t first = mode == 1 ? count : cn_overlap_first_count(count, mode);
        std::cout << "[CUDA CN stagger] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | mode=" << (mode == 1 ? "single" : "staggered")
                  << " | batch=" << count;
        if (mode != 1) std::cout << " | partitions=" << first << '+' << (count - first);
        std::cout << " | loop-overlap=no | generic=yes\n";
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
