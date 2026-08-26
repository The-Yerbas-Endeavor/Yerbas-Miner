// Unified production GPU autotuner.
//
// This is the only production-facing tuning policy. It deliberately tunes a
// small number of end-to-end GhostRider policies instead of stacking separate
// batch/CN/core micro-tuners and caches. Hash kernels and parity validation are
// unchanged.

constexpr int kCoreAlgorithmCount = 15;
constexpr int kCoreDefaultThreads = 256;
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

namespace {

constexpr int kGpuPolicyRevision = 2;

struct GpuPolicyShape {
    std::size_t batch{0};
    int core_threads{128};
    int cn_setup{64};
    int cn_loop{64};
    int cn_final{64};
};

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
        << "-cap" << capacity
        << "-drv" << driver_version
        << "-rt" << runtime_version << ".txt";
    return cn_cache_directory() / key.str();
}

bool valid_policy_threads(int value, int kernel_max, const cudaDeviceProp& props)
{
    return value == 64 || value == 128 || value == 256 || value == 512
        ? value <= kernel_max && value <= props.maxThreadsPerBlock
        : false;
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
        g.aes_backend = 0; // portable AES is the clean production baseline.
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
        !valid_cn_threads(shape.cn_setup, props.maxThreadsPerBlock) ||
        !valid_cn_threads(shape.cn_loop, props.maxThreadsPerBlock) ||
        !valid_cn_threads(shape.cn_final, props.maxThreadsPerBlock)) return false;

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
        std::ofstream out(path, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_GPU_POLICY " << kGpuPolicyRevision << ' '
            << shape.batch << ' ' << shape.core_threads << ' '
            << shape.cn_setup << ' ' << shape.cn_loop << ' ' << shape.cn_final << ' '
            << std::setprecision(12) << hps << '\n';
    } catch (...) {}
}

DeviceJob make_tune_job()
{
    DeviceJob job{};
    std::array<std::uint8_t, 80> header{};
    header[0] = 4;
    for (std::size_t i = 4; i < 76; ++i)
        header[i] = static_cast<std::uint8_t>((i * 29U + 0x53U) & 0xffU);
    std::copy(header.begin(), header.end(), std::begin(job.header));
    std::fill(std::begin(job.target_le), std::end(job.target_le), 0xffU);
    const auto schedule = ghostrider::stage_schedule_quiet(
        ghostrider::Work{header.data(), header.size()});
    std::copy(schedule.begin(), schedule.end(), std::begin(job.stages));
    return job;
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
    const int utility_threads = 256;
    const int utility_blocks = static_cast<int>((shape.batch + utility_threads - 1U) /
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
                                         scratchpads, contexts,
                                         policy.cn_geometries[algorithm]);
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

float median_policy_ms(cudaStream_t stream,
                       DeviceJob* d_job,
                       std::uint32_t* d_nonces,
                       std::uint8_t* states,
                       std::uint8_t* scratchpads,
                       cryptonight::SplitContext* contexts,
                       const DeviceJob& host_job,
                       const GpuPolicyShape& shape)
{
    std::array<float, 3> samples{};
    for (auto& ms : samples)
        ms = run_policy_once(stream, d_job, d_nonces, states, scratchpads, contexts,
                             host_job, shape);
    std::sort(samples.begin(), samples.end());
    return samples[1];
}

bool parity_policy(cudaStream_t stream,
                   DeviceJob* d_job,
                   std::uint32_t* d_nonces,
                   std::uint8_t* states,
                   std::uint8_t* scratchpads,
                   cryptonight::SplitContext* contexts,
                   const DeviceJob& host_job,
                   GpuPolicyShape baseline,
                   GpuPolicyShape candidate)
{
    baseline.batch = 2;
    candidate.batch = 2;
    std::array<std::uint8_t, 2 * kStateBytes> a{};
    std::array<std::uint8_t, 2 * kStateBytes> b{};
    (void)run_policy_once(stream, d_job, d_nonces, states, scratchpads, contexts, host_job, baseline);
    check_cuda(cudaMemcpyAsync(a.data(), states, a.size(), cudaMemcpyDeviceToHost, stream),
               "GPU policy baseline parity copy failed");
    check_cuda(cudaStreamSynchronize(stream), "GPU policy baseline parity sync failed");
    (void)run_policy_once(stream, d_job, d_nonces, states, scratchpads, contexts, host_job, candidate);
    check_cuda(cudaMemcpyAsync(b.data(), states, b.size(), cudaMemcpyDeviceToHost, stream),
               "GPU policy candidate parity copy failed");
    check_cuda(cudaStreamSynchronize(stream), "GPU policy candidate parity sync failed");
    return a == b;
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
    const int kernel_max = std::max(1, attrs.maxThreadsPerBlock);

    GpuTunePolicy cached{};
    if (load_gpu_policy_cache(device_id, props, capacity, kernel_max, cached))
        return cached;

    int default_core = 64;
    for (const int candidate : {256, 128, 64}) {
        if (valid_policy_threads(candidate, kernel_max, props)) {
            default_core = candidate;
            break;
        }
    }

    GpuPolicyShape baseline{};
    baseline.batch = std::max<std::size_t>(1U, std::min(requested_batch, capacity));
    baseline.core_threads = default_core;

    DeviceJob host_job = make_tune_job();
    DeviceJob* d_job = nullptr;
    std::uint32_t* d_nonces = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)),
               "cudaMalloc GPU policy job failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces), capacity * sizeof(std::uint32_t)),
                   "cudaMalloc GPU policy nonces failed");
        check_cuda(cudaMemcpyAsync(d_job, &host_job, sizeof(host_job), cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync GPU policy job failed");
        check_cuda(cudaStreamSynchronize(stream), "GPU policy job sync failed");

        GpuPolicyShape best = baseline;
        float best_ms = median_policy_ms(stream, d_job, d_nonces, states, scratchpads,
                                         contexts, host_job, baseline);

        std::vector<std::size_t> batch_candidates;
        for (const std::size_t raw : {baseline.batch,
                                      std::max<std::size_t>(1U, baseline.batch * 3U / 4U),
                                      std::max<std::size_t>(1U, baseline.batch / 2U)}) {
            std::size_t value = raw;
            if (value >= 256U) value = (value / 256U) * 256U;
            value = std::max<std::size_t>(1U, std::min(value, capacity));
            if (std::find(batch_candidates.begin(), batch_candidates.end(), value) == batch_candidates.end())
                batch_candidates.push_back(value);
        }
        for (const auto batch : batch_candidates) {
            GpuPolicyShape candidate = best;
            candidate.batch = batch;
            const float ms = median_policy_ms(stream, d_job, d_nonces, states, scratchpads,
                                              contexts, host_job, candidate);
            const double hps = ms > 0.0F ? static_cast<double>(batch) * 1000.0 / ms : 0.0;
            const double best_hps = best_ms > 0.0F ? static_cast<double>(best.batch) * 1000.0 / best_ms : 0.0;
            if (hps > best_hps) { best = candidate; best_ms = ms; }
        }

        for (const int threads : {64, 128, 256, 512}) {
            if (!valid_policy_threads(threads, kernel_max, props)) continue;
            GpuPolicyShape candidate = best;
            candidate.core_threads = threads;
            const float ms = median_policy_ms(stream, d_job, d_nonces, states, scratchpads,
                                              contexts, host_job, candidate);
            if (ms < best_ms) { best = candidate; best_ms = ms; }
        }

        constexpr std::array<std::array<int, 3>, 4> cn_profiles{{
            {{64, 64, 64}}, {{64, 32, 64}}, {{128, 64, 128}}, {{128, 128, 128}}
        }};
        for (const auto& profile : cn_profiles) {
            if (!valid_cn_threads(profile[0], props.maxThreadsPerBlock) ||
                !valid_cn_threads(profile[1], props.maxThreadsPerBlock) ||
                !valid_cn_threads(profile[2], props.maxThreadsPerBlock)) continue;
            GpuPolicyShape candidate = best;
            candidate.cn_setup = profile[0];
            candidate.cn_loop = profile[1];
            candidate.cn_final = profile[2];
            const float ms = median_policy_ms(stream, d_job, d_nonces, states, scratchpads,
                                              contexts, host_job, candidate);
            if (ms < best_ms) { best = candidate; best_ms = ms; }
        }

        const bool parity = parity_policy(stream, d_job, d_nonces, states, scratchpads,
                                          contexts, host_job, baseline, best);
        if (!parity) {
            std::cerr << "[GPU " << device_id << "] production GPU policy parity FAIL; baseline retained\n";
            best = baseline;
            best_ms = median_policy_ms(stream, d_job, d_nonces, states, scratchpads,
                                       contexts, host_job, best);
        }

        GpuTunePolicy policy{};
        apply_shape(policy, best);
        policy.throughput_hps = best_ms > 0.0F
            ? static_cast<double>(best.batch) * 1000.0 / best_ms : 0.0;
        policy.from_cache = false;
        save_gpu_policy_cache(device_id, props, capacity, best, policy.throughput_hps);

        std::cout << "[GPU " << device_id << "] production GPU autotune selected"
                  << " | batch=" << best.batch
                  << " | core=" << best.core_threads
                  << " | CN=" << best.cn_setup << '/' << best.cn_loop << '/' << best.cn_final
                  << " | parity=" << (parity ? "PASS" : "baseline")
                  << " | throughput=" << std::fixed << std::setprecision(2)
                  << policy.throughput_hps << " H/s"
                  << " | policy=gpu-v2" << std::defaultfloat << '\n';

        cudaFree(d_nonces);
        cudaFree(d_job);
        return policy;
    } catch (...) {
        if (d_nonces) cudaFree(d_nonces);
        if (d_job) cudaFree(d_job);
        throw;
    }
}
