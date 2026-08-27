// Instant-start production GPU policy + explicit bounded calibration.
//
// Normal production startup never benchmarks. It loads a previously calibrated
// policy when available, otherwise derives a conservative legal policy from
// runtime CUDA limits and starts immediately. --gpu-autotune enables a short
// one-shot whole-GhostRider calibration, exact parity-checks the winner, saves
// it per device/driver/runtime, and then mining continues with that policy.

constexpr int kCoreAlgorithmCount = 15;
constexpr int kCoreDefaultThreads = 256;
constexpr int kGpuCalibrationRevision = 1;
constexpr double kGpuCalibrationMinimumGain = 1.01;
static constexpr std::array<const char*, kCoreAlgorithmCount> kCoreAlgorithmNames{{
    "BLAKE-512", "BMW-512", "Groestl-512", "JH-512", "Keccak-512",
    "Skein-512", "Luffa-512", "CubeHash-512", "Shavite-512", "SIMD-512",
    "Echo-512", "Hamsi-512", "Fugue-512", "Shabal-512", "Whirlpool"
}};

struct GpuTunePolicy {
    std::size_t batch_size{0};
    std::array<CnGeometry, 6> cn_geometries{};
    std::array<int, kCoreAlgorithmCount> core_threads{};
    double throughput_hps{0.0};
    bool from_cache{false};
};

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& geometry)
{
    launch_split_cryptonight_variant_production<VariantIndex>(stream, states, count,
                                                              scratchpads, contexts,
                                                              geometry);
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

namespace {

struct GpuCalibrationShape {
    std::size_t batch{0};
    int core_threads{0};
    int cn_threads{0};
};

int faststart_warp_aligned_down(int value, int warp)
{
    warp = std::max(1, warp);
    return (value / warp) * warp;
}

int faststart_core_threads(const cudaDeviceProp& props, int kernel_max)
{
    const int warp = std::max(1, props.warpSize);
    const int limit = std::max(warp, std::min(kernel_max, props.maxThreadsPerBlock));
    return std::max(warp, faststart_warp_aligned_down(std::min(kCoreDefaultThreads, limit), warp));
}

bool gpu_calibration_requested()
{
    const char* value = std::getenv("YERBAS_GPU_AUTOTUNE");
    return value && *value && std::string(value) != "0";
}

std::filesystem::path gpu_calibration_cache_path(int device_id,
                                                 const cudaDeviceProp& props,
                                                 std::size_t capacity)
{
    int driver = 0;
    int runtime = 0;
    if (cudaDriverGetVersion(&driver) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    std::ostringstream name;
    name << "gpu-calibration-rev" << kGpuCalibrationRevision
         << "-dev" << device_id
         << '-' << sanitize_cache_component(props.name)
         << "-cc" << props.major << props.minor
         << "-sm" << props.multiProcessorCount
         << "-warp" << props.warpSize
         << "-mem" << props.totalGlobalMem
         << "-cap" << capacity
         << "-drv" << driver
         << "-rt" << runtime << ".txt";
    return cn_cache_directory() / name.str();
}

void apply_calibration_shape(GpuTunePolicy& policy, const GpuCalibrationShape& shape)
{
    policy.batch_size = shape.batch;
    policy.core_threads.fill(shape.core_threads);
    for (auto& geometry : policy.cn_geometries) {
        geometry = CnGeometry{};
        geometry.setup_threads = shape.cn_threads;
        geometry.loop_threads = shape.cn_threads;
        geometry.final_threads = shape.cn_threads;
        geometry.aes_backend = 0;
    }
}

bool load_gpu_calibration_cache(int device_id,
                                const cudaDeviceProp& props,
                                std::size_t capacity,
                                int kernel_max,
                                GpuTunePolicy& policy)
{
    const auto path = gpu_calibration_cache_path(device_id, props, capacity);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    GpuCalibrationShape shape{};
    double hps = 0.0;
    if (!(in >> magic >> revision >> shape.batch >> shape.core_threads >> shape.cn_threads >> hps))
        return false;
    const int warp = std::max(1, props.warpSize);
    if (magic != "YERBAS_GPU_CALIBRATION" || revision != kGpuCalibrationRevision ||
        shape.batch == 0 || shape.batch > capacity ||
        shape.core_threads < warp || shape.core_threads > kernel_max ||
        shape.core_threads > props.maxThreadsPerBlock || (shape.core_threads % warp) != 0 ||
        shape.cn_threads < warp || shape.cn_threads > props.maxThreadsPerBlock ||
        (shape.cn_threads % warp) != 0)
        return false;

    apply_calibration_shape(policy, shape);
    policy.throughput_hps = hps;
    policy.from_cache = true;
    std::cout << "[GPU " << device_id << "] GPU calibration cache loaded"
              << " | batch=" << shape.batch
              << " | core=" << shape.core_threads
              << " | CN=" << shape.cn_threads << '/' << shape.cn_threads << '/' << shape.cn_threads
              << " | throughput=" << std::fixed << std::setprecision(2) << hps << " H/s"
              << " | startup-benchmark=off" << std::defaultfloat << '\n';
    return true;
}

void save_gpu_calibration_cache(int device_id,
                                const cudaDeviceProp& props,
                                std::size_t capacity,
                                const GpuCalibrationShape& shape,
                                double hps)
{
    try {
        const auto path = gpu_calibration_cache_path(device_id, props, capacity);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_GPU_CALIBRATION " << kGpuCalibrationRevision << ' '
            << shape.batch << ' ' << shape.core_threads << ' ' << shape.cn_threads << ' '
            << std::setprecision(12) << hps << '\n';
        out.close();
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] GPU calibration cache write skipped: " << e.what() << '\n';
    }
}

DeviceJob make_gpu_calibration_job()
{
    DeviceJob best{};
    std::uint64_t best_score = 0;
    for (unsigned int sample = 0; sample < 24U; ++sample) {
        std::array<std::uint8_t, 80> header{};
        header[0] = 4;
        for (std::size_t i = 4; i < 76; ++i)
            header[i] = static_cast<std::uint8_t>((i * (19U + sample * 7U) + 0x4dU + sample * 31U) & 0xffU);
        const auto schedule = ghostrider::stage_schedule_quiet(ghostrider::Work{header.data(), header.size()});
        std::uint64_t score = 0;
        for (const std::uint8_t encoded : schedule) {
            if ((encoded & ghostrider::kCryptoNightStageFlag) == 0) continue;
            const std::uint8_t variant = static_cast<std::uint8_t>(encoded & 0x7fU);
            if (variant >= cryptonight::kVariantConfigs.size()) continue;
            const auto& cfg = cryptonight::kVariantConfigs[variant];
            score += static_cast<std::uint64_t>(cfg.page_size) +
                     static_cast<std::uint64_t>(cfg.iterations) * 4ULL;
        }
        if (score <= best_score) continue;
        best_score = score;
        std::copy(header.begin(), header.end(), std::begin(best.header));
        std::fill(std::begin(best.target_le), std::end(best.target_le), 0xffU);
        std::copy(schedule.begin(), schedule.end(), std::begin(best.stages));
    }
    return best;
}

std::string gpu_calibration_cn_summary(const DeviceJob& job)
{
    std::ostringstream out;
    bool first = true;
    for (const std::uint8_t encoded : job.stages) {
        if ((encoded & ghostrider::kCryptoNightStageFlag) == 0) continue;
        if (!first) out << '/';
        first = false;
        out << ghostrider::cryptonight_name(static_cast<std::uint8_t>(encoded & 0x7fU));
    }
    return out.str();
}

float run_gpu_calibration_once(cudaStream_t stream,
                               DeviceJob* d_job,
                               std::uint32_t* d_nonces,
                               std::uint8_t* states,
                               std::uint8_t* scratchpads,
                               cryptonight::SplitContext* contexts,
                               const DeviceJob& job,
                               const GpuCalibrationShape& shape)
{
    check_cuda(cudaMemcpyAsync(d_job, &job, sizeof(job), cudaMemcpyHostToDevice, stream),
               "GPU calibration job upload failed");
    const int utility_blocks = static_cast<int>((shape.batch + static_cast<std::size_t>(shape.core_threads) - 1U) /
                                                static_cast<std::size_t>(shape.core_threads));
    initialize_nonce_batch<<<utility_blocks, shape.core_threads, 0, stream>>>(
        0x6a000000U, d_nonces, shape.batch);
    check_cuda(cudaGetLastError(), "GPU calibration nonce initialization failed");

    GpuTunePolicy policy{};
    apply_calibration_shape(policy, shape);
    cudaEvent_t begin{};
    cudaEvent_t end{};
    check_cuda(cudaEventCreate(&begin), "GPU calibration event create failed");
    check_cuda(cudaEventCreate(&end), "GPU calibration event create failed");
    try {
        check_cuda(cudaEventRecord(begin, stream), "GPU calibration event record failed");
        for (int stage_index = 0; stage_index < 18; ++stage_index) {
            const std::uint8_t encoded = job.stages[static_cast<std::size_t>(stage_index)];
            const bool is_cn = (encoded & ghostrider::kCryptoNightStageFlag) != 0;
            const std::uint8_t algorithm = static_cast<std::uint8_t>(encoded & 0x7fU);
            if (is_cn) {
                launch_split_cryptonight(stream, states, shape.batch, algorithm,
                                         scratchpads, contexts, policy.cn_geometries[algorithm]);
            } else {
                const int blocks = static_cast<int>((shape.batch + static_cast<std::size_t>(shape.core_threads) - 1U) /
                                                    static_cast<std::size_t>(shape.core_threads));
                core512_stage<<<blocks, shape.core_threads, 0, stream>>>(
                    d_job, d_nonces, states, shape.batch, stage_index, algorithm);
                check_cuda(cudaGetLastError(), "GPU calibration core launch failed");
            }
        }
        check_cuda(cudaEventRecord(end, stream), "GPU calibration event record failed");
        check_cuda(cudaEventSynchronize(end), "GPU calibration event sync failed");
        float ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&ms, begin, end), "GPU calibration elapsed time failed");
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
        return ms;
    } catch (...) {
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
        throw;
    }
}

double gpu_calibration_hps(cudaStream_t stream,
                           DeviceJob* d_job,
                           std::uint32_t* d_nonces,
                           std::uint8_t* states,
                           std::uint8_t* scratchpads,
                           cryptonight::SplitContext* contexts,
                           const DeviceJob& job,
                           const GpuCalibrationShape& shape)
{
    const float ms = run_gpu_calibration_once(stream, d_job, d_nonces, states,
                                              scratchpads, contexts, job, shape);
    return ms > 0.0F ? static_cast<double>(shape.batch) * 1000.0 / static_cast<double>(ms) : 0.0;
}

bool gpu_calibration_parity(cudaStream_t stream,
                            DeviceJob* d_job,
                            std::uint32_t* d_nonces,
                            std::uint8_t* states,
                            std::uint8_t* scratchpads,
                            cryptonight::SplitContext* contexts,
                            const DeviceJob& job,
                            GpuCalibrationShape baseline,
                            GpuCalibrationShape candidate)
{
    baseline.batch = 2;
    candidate.batch = 2;
    std::array<std::uint8_t, 2 * kStateBytes> a{};
    std::array<std::uint8_t, 2 * kStateBytes> b{};
    (void)run_gpu_calibration_once(stream, d_job, d_nonces, states, scratchpads, contexts, job, baseline);
    check_cuda(cudaMemcpyAsync(a.data(), states, a.size(), cudaMemcpyDeviceToHost, stream),
               "GPU calibration baseline parity copy failed");
    check_cuda(cudaStreamSynchronize(stream), "GPU calibration baseline parity sync failed");
    (void)run_gpu_calibration_once(stream, d_job, d_nonces, states, scratchpads, contexts, job, candidate);
    check_cuda(cudaMemcpyAsync(b.data(), states, b.size(), cudaMemcpyDeviceToHost, stream),
               "GPU calibration candidate parity copy failed");
    check_cuda(cudaStreamSynchronize(stream), "GPU calibration candidate parity sync failed");
    return a == b;
}

GpuTunePolicy run_gpu_calibration(int device_id,
                                  const cudaDeviceProp& props,
                                  int kernel_max,
                                  std::size_t requested_batch,
                                  std::size_t capacity,
                                  cudaStream_t stream,
                                  std::uint8_t* states,
                                  std::uint8_t* scratchpads,
                                  cryptonight::SplitContext* contexts)
{
    const int warp = std::max(1, props.warpSize);
    GpuCalibrationShape baseline{};
    baseline.batch = std::max<std::size_t>(1U, std::min(requested_batch, capacity));
    baseline.core_threads = faststart_core_threads(props, kernel_max);
    baseline.cn_threads = warp;

    DeviceJob* d_job = nullptr;
    std::uint32_t* d_nonces = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)),
               "cudaMalloc GPU calibration job failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces), capacity * sizeof(std::uint32_t)),
                   "cudaMalloc GPU calibration nonces failed");
        const DeviceJob job = make_gpu_calibration_job();
        GpuCalibrationShape best = baseline;

        constexpr unsigned int kMaxCandidateMeasurements = 5U; // baseline + four alternatives
        unsigned int progress = 1U;
        std::cout << "[GPU " << device_id << "] GPU autotune calibration starting"
                  << " | candidates=" << kMaxCandidateMeasurements
                  << " | metric=one heavy whole-GhostRider schedule"
                  << " | CN=" << gpu_calibration_cn_summary(job) << '\n';
        std::cout << "[GPU " << device_id << "] GPU autotune 1/" << kMaxCandidateMeasurements
                  << " | baseline"
                  << " | batch=" << baseline.batch
                  << " | core=" << baseline.core_threads
                  << " | CN=" << baseline.cn_threads
                  << " | testing..." << std::flush;
        double best_hps = gpu_calibration_hps(stream, d_job, d_nonces, states,
                                              scratchpads, contexts, job, best);
        std::cout << std::fixed << std::setprecision(2)
                  << " done | " << best_hps << " H/s | best=" << best_hps << " H/s"
                  << std::defaultfloat << '\n';

        auto test = [&](GpuCalibrationShape candidate, const char* label) {
            if (candidate.batch == best.batch && candidate.core_threads == best.core_threads &&
                candidate.cn_threads == best.cn_threads) return;
            ++progress;
            std::cout << "[GPU " << device_id << "] GPU autotune " << progress << '/'
                      << kMaxCandidateMeasurements << " | " << label
                      << " | batch=" << candidate.batch
                      << " | core=" << candidate.core_threads
                      << " | CN=" << candidate.cn_threads
                      << " | testing..." << std::flush;
            try {
                const double previous_best = best_hps;
                const double hps = gpu_calibration_hps(stream, d_job, d_nonces, states,
                                                       scratchpads, contexts, job, candidate);
                const double gain_pct = previous_best > 0.0 ? (hps / previous_best - 1.0) * 100.0 : 0.0;
                const bool wins = hps > previous_best * kGpuCalibrationMinimumGain;
                std::cout << std::fixed << std::setprecision(2)
                          << " done | " << hps << " H/s"
                          << " | gain=" << (gain_pct >= 0.0 ? "+" : "") << gain_pct << '%'
                          << " | gate=+" << ((kGpuCalibrationMinimumGain - 1.0) * 100.0) << '%'
                          << " | " << (wins ? "NEW BEST" : "keep current")
                          << " | best=" << (wins ? hps : previous_best) << " H/s"
                          << std::defaultfloat << '\n';
                if (wins) { best = candidate; best_hps = hps; }
            } catch (const std::exception& e) {
                cudaGetLastError();
                std::cerr << " skipped | " << e.what() << '\n';
            }
        };

        if (baseline.batch >= 4U) {
            GpuCalibrationShape candidate = best;
            candidate.batch = std::max<std::size_t>(1U, (baseline.batch * 7U) / 8U);
            test(candidate, "batch-7/8");
            candidate = best;
            candidate.batch = std::max<std::size_t>(1U, (baseline.batch * 3U) / 4U);
            test(candidate, "batch-3/4");
        }

        const int half_core = std::max(warp, faststart_warp_aligned_down(baseline.core_threads / 2, warp));
        if (half_core != baseline.core_threads) {
            GpuCalibrationShape candidate = best;
            candidate.core_threads = half_core;
            test(candidate, "core-half");
        }

        const int wider_cn = std::min(props.maxThreadsPerBlock, warp * 2);
        if (wider_cn != warp && (wider_cn % warp) == 0) {
            GpuCalibrationShape candidate = best;
            candidate.cn_threads = wider_cn;
            test(candidate, "cn-wide");
        }

        std::cout << "[GPU " << device_id << "] GPU autotune validation"
                  << " | candidate=" << best.batch << '/' << best.core_threads << '/' << best.cn_threads
                  << " | exact parity..." << std::flush;
        const bool parity = gpu_calibration_parity(stream, d_job, d_nonces, states,
                                                   scratchpads, contexts, job, baseline, best);
        std::cout << " done | parity=" << (parity ? "PASS" : "FAIL") << '\n';
        if (!parity) {
            std::cerr << "[GPU " << device_id << "] GPU autotune parity FAIL; safe baseline retained\n";
            best = baseline;
            best_hps = gpu_calibration_hps(stream, d_job, d_nonces, states,
                                           scratchpads, contexts, job, best);
        }

        std::cout << "[GPU " << device_id << "] GPU autotune saving calibration cache..." << std::flush;
        save_gpu_calibration_cache(device_id, props, capacity, best, best_hps);
        std::cout << " done\n";
        GpuTunePolicy policy{};
        apply_calibration_shape(policy, best);
        policy.throughput_hps = best_hps;
        policy.from_cache = false;

        std::cout << "[GPU " << device_id << "] GPU autotune selected"
                  << " | batch=" << best.batch
                  << " | core=" << best.core_threads
                  << " | CN=" << best.cn_threads << '/' << best.cn_threads << '/' << best.cn_threads
                  << " | parity=" << (parity ? "PASS" : "baseline")
                  << " | throughput=" << std::fixed << std::setprecision(2) << best_hps << " H/s"
                  << " | cache=gpu-calibration-v" << kGpuCalibrationRevision
                  << std::defaultfloat << '\n';

        cudaFree(d_nonces);
        cudaFree(d_job);
        return policy;
    } catch (...) {
        if (d_nonces) cudaFree(d_nonces);
        if (d_job) cudaFree(d_job);
        throw;
    }
}

} // anonymous namespace

GpuTunePolicy build_gpu_tune_policy(int device_id,
                                    std::size_t requested_batch,
                                    std::size_t capacity,
                                    cudaStream_t stream,
                                    std::uint8_t* states,
                                    std::uint8_t* scratchpads,
                                    cryptonight::SplitContext* contexts)
{
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties GPU fast-start failed");

    cudaFuncAttributes attrs{};
    check_cuda(cudaFuncGetAttributes(&attrs, core512_stage),
               "cudaFuncGetAttributes GPU fast-start failed");

    const int kernel_max = std::max(1, std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock));

    if (gpu_calibration_requested()) {
        return run_gpu_calibration(device_id, props, kernel_max, requested_batch, capacity,
                                   stream, states, scratchpads, contexts);
    }

    GpuTunePolicy cached{};
    if (load_gpu_calibration_cache(device_id, props, capacity, kernel_max, cached))
        return cached;

    const int core_threads = faststart_core_threads(props, kernel_max);
    const int warp = std::max(1, props.warpSize);
    GpuTunePolicy policy{};
    policy.batch_size = std::max<std::size_t>(1U, std::min(requested_batch, capacity));
    policy.core_threads.fill(core_threads);
    for (auto& geometry : policy.cn_geometries) {
        geometry = CnGeometry{};
        geometry.setup_threads = warp;
        geometry.loop_threads = warp;
        geometry.final_threads = warp;
        geometry.aes_backend = 0;
    }

    std::cout << "[GPU " << device_id << "] production GPU fast-start"
              << " | batch=" << policy.batch_size
              << " | core=" << core_threads
              << " | CN=" << warp << '/' << warp << '/' << warp
              << " | source=runtime-safe"
              << " | startup-benchmark=off\n";
    return policy;
}
