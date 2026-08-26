// Unified production GPU autotuner.
//
// One policy is selected per CUDA device by measuring complete GhostRider
// pipelines. Candidate launch widths are derived from the device warp size and
// compiled kernel limits. No GPU model, compute capability, SM count or memory
// size is hard-coded as a selection rule.

constexpr int kCoreAlgorithmCount = 15;
constexpr int kCoreDefaultThreads = 256; // fallback only; tuner derives baseline
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

// Clean production dispatcher: use the proven base split-CryptoNight kernels,
// with geometry supplied by the single GPU policy. Historical loop/dual/coop
// selectors are not part of the production path anymore.
template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& geometry)
{
    launch_split_cryptonight_variant_legacy<VariantIndex>(stream, states, count,
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

constexpr int kGpuPolicyRevision = 3;
constexpr double kPolicyMinimumGain = 1.01; // ignore sub-1% tuning noise

struct GpuPolicyShape {
    std::size_t batch{0};
    int core_threads{0};
    int cn_setup{0};
    int cn_loop{0};
    int cn_final{0};
};

int warp_aligned_down(int value, int warp)
{
    warp = std::max(1, warp);
    return (value / warp) * warp;
}

bool valid_policy_threads(int value, int limit, const cudaDeviceProp& props)
{
    const int warp = std::max(1, props.warpSize);
    return value >= warp && value <= limit && value <= props.maxThreadsPerBlock &&
           (value % warp) == 0;
}

int runtime_baseline_threads(const cudaDeviceProp& props, int kernel_max)
{
    const int warp = std::max(1, props.warpSize);
    const int limit = std::max(warp, std::min(kernel_max, props.maxThreadsPerBlock));
    const int preferred = std::min(kCoreDefaultThreads, limit);
    return std::max(warp, warp_aligned_down(preferred, warp));
}

std::vector<int> runtime_thread_candidates(const cudaDeviceProp& props, int kernel_max)
{
    const int warp = std::max(1, props.warpSize);
    const int limit = std::max(warp, std::min(kernel_max, props.maxThreadsPerBlock));
    std::vector<int> out;
    auto add = [&](int raw) {
        const int value = std::max(warp, warp_aligned_down(std::min(raw, limit), warp));
        if (!valid_policy_threads(value, limit, props)) return;
        if (std::find(out.begin(), out.end(), value) == out.end()) out.push_back(value);
    };
    add(warp);
    add(warp * 2);
    add(warp * 4);
    add(warp * 8);
    add(warp * 16);
    add(runtime_baseline_threads(props, limit));
    add(limit);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::size_t> runtime_batch_candidates(const cudaDeviceProp& props,
                                                  std::size_t requested,
                                                  std::size_t capacity)
{
    const std::size_t warp = static_cast<std::size_t>(std::max(1, props.warpSize));
    const std::size_t sms = static_cast<std::size_t>(std::max(1, props.multiProcessorCount));
    const std::size_t quantum = std::max<std::size_t>(warp, warp * sms);
    std::vector<std::size_t> out;
    auto add = [&](std::size_t raw) {
        if (capacity == 0) return;
        raw = std::max<std::size_t>(1U, std::min(raw, capacity));
        if (raw >= quantum) raw = (raw / quantum) * quantum;
        raw = std::max<std::size_t>(1U, std::min(raw, capacity));
        if (std::find(out.begin(), out.end(), raw) == out.end()) out.push_back(raw);
    };

    add(requested);
    add(requested / 2U);
    add((requested * 3U) / 4U);
    add((requested * 5U) / 4U);
    add((capacity * 3U) / 4U);
    add(capacity);
    std::sort(out.begin(), out.end());
    return out;
}

std::filesystem::path gpu_policy_cache_path(int device_id,
                                            const cudaDeviceProp& props,
                                            std::size_t capacity)
{
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    std::ostringstream key;
    key << "gpu-policy-rev" << kGpuPolicyRevision
        << "-dev" << device_id
        << '-' << sanitize_cache_component(props.name)
        << "-cc" << props.major << props.minor
        << "-sm" << props.multiProcessorCount
        << "-warp" << props.warpSize
        << "-mem" << props.totalGlobalMem
        << "-cap" << capacity
        << "-drv" << driver_version
        << "-rt" << runtime_version << ".txt";
    return cn_cache_directory() / key.str();
}

void apply_shape(GpuTunePolicy& policy, const GpuPolicyShape& shape)
{
    policy.batch_size = shape.batch;
    policy.core_threads.fill(shape.core_threads);
    for (auto& g : policy.cn_geometries) {
        g = CnGeometry{};
        g.setup_threads = shape.cn_setup;
        g.loop_threads = shape.cn_loop;
        g.final_threads = shape.cn_final;
        g.aes_backend = 0; // portable AES is the generic correctness baseline
    }
}

bool load_gpu_policy_cache(int device_id,
                           const cudaDeviceProp& props,
                           std::size_t capacity,
                           int kernel_max,
                           GpuTunePolicy& out)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = gpu_policy_cache_path(device_id, props, capacity);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    GpuPolicyShape shape{};
    double hps = 0.0;
    if (!(in >> magic >> revision >> shape.batch >> shape.core_threads
             >> shape.cn_setup >> shape.cn_loop >> shape.cn_final >> hps)) return false;
    if (magic != "YERBAS_GPU_POLICY" || revision != kGpuPolicyRevision ||
        shape.batch == 0 || shape.batch > capacity ||
        !valid_policy_threads(shape.core_threads, kernel_max, props) ||
        !valid_policy_threads(shape.cn_setup, props.maxThreadsPerBlock, props) ||
        !valid_policy_threads(shape.cn_loop, props.maxThreadsPerBlock, props) ||
        !valid_policy_threads(shape.cn_final, props.maxThreadsPerBlock, props)) return false;

    apply_shape(out, shape);
    out.throughput_hps = hps;
    out.from_cache = true;
    std::cout << "[GPU " << device_id << "] production GPU policy cache loaded"
              << " | batch=" << shape.batch
              << " | core=" << shape.core_threads
              << " | CN=" << shape.cn_setup << '/' << shape.cn_loop << '/' << shape.cn_final
              << " | throughput=" << std::fixed << std::setprecision(2) << hps << " H/s"
              << std::defaultfloat << '\n';
    return true;
}

void save_gpu_policy_cache(int device_id,
                           const cudaDeviceProp& props,
                           std::size_t capacity,
                           const GpuPolicyShape& shape,
                           double hps)
{
    try {
        const auto path = gpu_policy_cache_path(device_id, props, capacity);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_GPU_POLICY " << kGpuPolicyRevision << ' '
            << shape.batch << ' ' << shape.core_threads << ' '
            << shape.cn_setup << ' ' << shape.cn_loop << ' ' << shape.cn_final << ' '
            << std::setprecision(12) << hps << '\n';
        out.close();
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (...) {}
}

std::array<DeviceJob, 3> make_tune_jobs()
{
    std::array<DeviceJob, 3> jobs{};
    for (std::size_t sample = 0; sample < jobs.size(); ++sample) {
        std::array<std::uint8_t, 80> header{};
        header[0] = 4;
        for (std::size_t i = 4; i < 76; ++i)
            header[i] = static_cast<std::uint8_t>((i * (23U + sample * 6U) + 0x35U + sample * 41U) & 0xffU);
        std::copy(header.begin(), header.end(), std::begin(jobs[sample].header));
        std::fill(std::begin(jobs[sample].target_le), std::end(jobs[sample].target_le), 0xffU);
        const auto schedule = ghostrider::stage_schedule_quiet(
            ghostrider::Work{header.data(), header.size()});
        std::copy(schedule.begin(), schedule.end(), std::begin(jobs[sample].stages));
    }
    return jobs;
}

float run_policy_once(cudaStream_t stream,
                      DeviceJob* d_job,
                      std::uint32_t* d_nonces,
                      std::uint8_t* states,
                      std::uint8_t* scratchpads,
                      cryptonight::SplitContext* contexts,
                      const DeviceJob& host_job,
                      const GpuPolicyShape& shape)
{
    check_cuda(cudaMemcpyAsync(d_job, &host_job, sizeof(host_job), cudaMemcpyHostToDevice, stream),
               "GPU policy job upload failed");
    const int utility_threads = shape.core_threads;
    const int utility_blocks = static_cast<int>((shape.batch + static_cast<std::size_t>(utility_threads) - 1U) /
                                                static_cast<std::size_t>(utility_threads));
    initialize_nonce_batch<<<utility_blocks, utility_threads, 0, stream>>>(
        0x62000000U, d_nonces, shape.batch);
    check_cuda(cudaGetLastError(), "GPU policy nonce initialization failed");

    GpuTunePolicy policy{};
    apply_shape(policy, shape);

    cudaEvent_t begin{};
    cudaEvent_t end{};
    check_cuda(cudaEventCreate(&begin), "GPU policy event create failed");
    check_cuda(cudaEventCreate(&end), "GPU policy event create failed");
    try {
        check_cuda(cudaEventRecord(begin, stream), "GPU policy event record failed");
        for (int stage_index = 0; stage_index < 18; ++stage_index) {
            const std::uint8_t stage = host_job.stages[static_cast<std::size_t>(stage_index)];
            const bool is_cn = (stage & ghostrider::kCryptoNightStageFlag) != 0;
            const std::uint8_t algorithm = static_cast<std::uint8_t>(stage & 0x7fU);
            if (is_cn) {
                launch_split_cryptonight(stream, states, shape.batch, algorithm,
                                         scratchpads, contexts, policy.cn_geometries[algorithm]);
            } else {
                const int blocks = static_cast<int>((shape.batch + static_cast<std::size_t>(shape.core_threads) - 1U) /
                                                    static_cast<std::size_t>(shape.core_threads));
                core512_stage<<<blocks, shape.core_threads, 0, stream>>>(
                    d_job, d_nonces, states, shape.batch, stage_index, algorithm);
                check_cuda(cudaGetLastError(), "GPU policy core launch failed");
            }
        }
        check_cuda(cudaEventRecord(end, stream), "GPU policy event record failed");
        check_cuda(cudaEventSynchronize(end), "GPU policy event sync failed");
        float ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&ms, begin, end), "GPU policy elapsed time failed");
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
        return ms;
    } catch (...) {
        cudaEventDestroy(end);
        cudaEventDestroy(begin);
        throw;
    }
}

double benchmark_policy_hps(cudaStream_t stream,
                            DeviceJob* d_job,
                            std::uint32_t* d_nonces,
                            std::uint8_t* states,
                            std::uint8_t* scratchpads,
                            cryptonight::SplitContext* contexts,
                            const std::array<DeviceJob, 3>& jobs,
                            const GpuPolicyShape& shape)
{
    std::array<double, 3> hps{};
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        std::array<float, 3> ms{};
        for (auto& value : ms)
            value = run_policy_once(stream, d_job, d_nonces, states, scratchpads,
                                    contexts, jobs[i], shape);
        std::sort(ms.begin(), ms.end());
        hps[i] = ms[1] > 0.0F ? static_cast<double>(shape.batch) * 1000.0 / ms[1] : 0.0;
    }
    std::sort(hps.begin(), hps.end());
    return hps[1];
}

bool parity_policy(cudaStream_t stream,
                   DeviceJob* d_job,
                   std::uint32_t* d_nonces,
                   std::uint8_t* states,
                   std::uint8_t* scratchpads,
                   cryptonight::SplitContext* contexts,
                   const std::array<DeviceJob, 3>& jobs,
                   GpuPolicyShape baseline,
                   GpuPolicyShape candidate)
{
    baseline.batch = 2;
    candidate.batch = 2;
    for (const auto& job : jobs) {
        std::array<std::uint8_t, 2 * kStateBytes> a{};
        std::array<std::uint8_t, 2 * kStateBytes> b{};
        (void)run_policy_once(stream, d_job, d_nonces, states, scratchpads, contexts, job, baseline);
        check_cuda(cudaMemcpyAsync(a.data(), states, a.size(), cudaMemcpyDeviceToHost, stream),
                   "GPU policy baseline parity copy failed");
        check_cuda(cudaStreamSynchronize(stream), "GPU policy baseline parity sync failed");
        (void)run_policy_once(stream, d_job, d_nonces, states, scratchpads, contexts, job, candidate);
        check_cuda(cudaMemcpyAsync(b.data(), states, b.size(), cudaMemcpyDeviceToHost, stream),
                   "GPU policy candidate parity copy failed");
        check_cuda(cudaStreamSynchronize(stream), "GPU policy candidate parity sync failed");
        if (a != b) return false;
    }
    return true;
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
               "cudaGetDeviceProperties GPU policy failed");

    cudaFuncAttributes attrs{};
    check_cuda(cudaFuncGetAttributes(&attrs, core512_stage),
               "cudaFuncGetAttributes GPU policy failed");
    const int kernel_max = std::max(1, std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock));
    const int baseline_threads = runtime_baseline_threads(props, kernel_max);
    const auto thread_candidates = runtime_thread_candidates(props, kernel_max);

    GpuTunePolicy cached{};
    if (load_gpu_policy_cache(device_id, props, capacity, kernel_max, cached))
        return cached;

    GpuPolicyShape baseline{};
    baseline.batch = std::max<std::size_t>(1U, std::min(requested_batch, capacity));
    baseline.core_threads = baseline_threads;
    baseline.cn_setup = baseline_threads;
    baseline.cn_loop = baseline_threads;
    baseline.cn_final = baseline_threads;

    const auto batch_candidates = runtime_batch_candidates(props, baseline.batch, capacity);
    const auto jobs = make_tune_jobs();

    DeviceJob* d_job = nullptr;
    std::uint32_t* d_nonces = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)),
               "cudaMalloc GPU policy job failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces), capacity * sizeof(std::uint32_t)),
                   "cudaMalloc GPU policy nonces failed");

        std::cout << "[GPU " << device_id << "] production GPU autotune starting"
                  << " | warp=" << props.warpSize
                  << " | kernel_max=" << kernel_max
                  << " | baseline_threads=" << baseline_threads
                  << " | thread_candidates=" << thread_candidates.size()
                  << " | batch_candidates=" << batch_candidates.size()
                  << " | schedules=" << jobs.size()
                  << " | metric=median full GhostRider H/s\n";

        GpuPolicyShape best = baseline;
        double best_hps = benchmark_policy_hps(stream, d_job, d_nonces, states,
                                               scratchpads, contexts, jobs, best);

        auto test_candidate = [&](const GpuPolicyShape& candidate, const char* label) {
            try {
                const double hps = benchmark_policy_hps(stream, d_job, d_nonces, states,
                                                        scratchpads, contexts, jobs, candidate);
                const bool wins = hps > best_hps * kPolicyMinimumGain;
                if (std::getenv("YERBAS_DIAGNOSTICS")) {
                    std::cout << std::fixed << std::setprecision(2)
                              << "[CUDA GPU tune] GPU " << device_id
                              << " | " << label
                              << " | batch=" << candidate.batch
                              << " | core=" << candidate.core_threads
                              << " | CN=" << candidate.cn_setup << '/'
                              << candidate.cn_loop << '/' << candidate.cn_final
                              << " | throughput=" << hps << " H/s"
                              << " | selected=" << (wins ? "yes" : "no")
                              << std::defaultfloat << '\n';
                }
                if (wins) {
                    best = candidate;
                    best_hps = hps;
                }
            } catch (const std::exception& e) {
                cudaGetLastError();
                if (std::getenv("YERBAS_DIAGNOSTICS"))
                    std::cerr << "[CUDA GPU tune] GPU " << device_id
                              << " | " << label << " skipped: " << e.what() << '\n';
            }
        };

        for (const auto batch : batch_candidates) {
            GpuPolicyShape candidate = best;
            candidate.batch = batch;
            test_candidate(candidate, "batch");
        }

        for (const int threads : thread_candidates) {
            GpuPolicyShape candidate = best;
            candidate.core_threads = threads;
            test_candidate(candidate, "core");
        }

        // Tune each CN phase independently, but judge every candidate by complete
        // GhostRider throughput across multiple schedules.
        for (const int threads : thread_candidates) {
            GpuPolicyShape candidate = best;
            candidate.cn_setup = threads;
            test_candidate(candidate, "cn-setup");
        }
        for (const int threads : thread_candidates) {
            GpuPolicyShape candidate = best;
            candidate.cn_loop = threads;
            test_candidate(candidate, "cn-loop");
        }
        for (const int threads : thread_candidates) {
            GpuPolicyShape candidate = best;
            candidate.cn_final = threads;
            test_candidate(candidate, "cn-final");
        }

        const bool parity = parity_policy(stream, d_job, d_nonces, states, scratchpads,
                                          contexts, jobs, baseline, best);
        if (!parity) {
            std::cerr << "[GPU " << device_id << "] production GPU policy parity FAIL; baseline retained\n";
            best = baseline;
            best_hps = benchmark_policy_hps(stream, d_job, d_nonces, states,
                                            scratchpads, contexts, jobs, best);
        }

        GpuTunePolicy policy{};
        apply_shape(policy, best);
        policy.throughput_hps = best_hps;
        policy.from_cache = false;
        save_gpu_policy_cache(device_id, props, capacity, best, best_hps);

        std::cout << "[GPU " << device_id << "] production GPU autotune selected"
                  << " | batch=" << best.batch
                  << " | core=" << best.core_threads
                  << " | CN=" << best.cn_setup << '/' << best.cn_loop << '/' << best.cn_final
                  << " | parity=" << (parity ? "PASS" : "baseline")
                  << " | throughput=" << std::fixed << std::setprecision(2)
                  << best_hps << " H/s"
                  << " | policy=gpu-v3" << std::defaultfloat << '\n';

        cudaFree(d_nonces);
        cudaFree(d_job);
        return policy;
    } catch (...) {
        if (d_nonces) cudaFree(d_nonces);
        if (d_job) cudaFree(d_job);
        throw;
    }
}
