struct BatchEngine::Impl {
    int device_id{-1};
    std::size_t batch_size{0};
    std::size_t batch_capacity{0};
    unsigned int fallback_threads{1};
    cudaStream_t stream{};
    DeviceJob* d_job{nullptr};
    std::uint32_t* d_nonces{nullptr};
    std::uint8_t* d_states{nullptr};
    std::uint8_t* d_cn_scratchpads{nullptr};
    cryptonight::SplitContext* d_cn_contexts{nullptr};
    DeviceCandidate* d_candidates{nullptr};
    unsigned int* d_candidate_count{nullptr};
    JobDescriptor host_job{};
    std::vector<std::uint8_t> host_states;
    std::array<CnGeometry, 6> cn_geometries{};
    std::uint64_t scan_count{0};
    std::uint64_t stage_profile_samples{0};
    std::array<double, 256> rolling_stage_ms{};
    std::array<std::uint64_t, 256> rolling_stage_count{};
    bool job_loaded{false};

    Impl(int id, std::size_t requested_size, unsigned int requested_fallback_threads)
        : device_id(id), batch_size(std::min(requested_size, cryptonight::kInitialMaxBatch)),
          batch_capacity(batch_size),
          fallback_threads(std::max(1u, requested_fallback_threads)), host_states(batch_capacity * kStateBytes)
    {
        if (batch_capacity == 0) throw std::runtime_error("CUDA batch size must be greater than zero");
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)), "cudaMalloc job failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces), batch_capacity * sizeof(std::uint32_t)), "cudaMalloc nonce batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_states), batch_capacity * kStateBytes), "cudaMalloc 512-bit state batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_scratchpads), batch_capacity * kCnScratchpadStride), "cudaMalloc CryptoNight scratchpads failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_contexts), batch_capacity * sizeof(cryptonight::SplitContext)), "cudaMalloc CryptoNight split contexts failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_candidates), kMaxCandidates * sizeof(DeviceCandidate)), "cudaMalloc candidate buffer failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_candidate_count), sizeof(unsigned int)), "cudaMalloc candidate counter failed");
        try {
            cn_geometries = autotune_cn_geometries(device_id, stream, d_states, batch_size,
                                                   d_cn_scratchpads, d_cn_contexts);
        } catch (const std::exception& e) {
            cn_geometries.fill(CnGeometry{});
            cudaGetLastError();
            std::cerr << "[GPU " << device_id << "] CryptoNight geometry autotune failed; using 64/64/64: "
                      << e.what() << '\n';
        }

        const char* active_batch_env = std::getenv("YERBAS_CUDA_ACTIVE_BATCH");
        if (active_batch_env && *active_batch_env) {
            try {
                const std::size_t requested_active = static_cast<std::size_t>(std::stoull(active_batch_env));
                if (requested_active > 0 && requested_active <= batch_capacity) {
                    batch_size = requested_active;
                    std::cout << "[GPU " << device_id << "] CUDA active batch override"
                              << " | active=" << batch_size
                              << " | capacity=" << batch_capacity << '\n';
                } else {
                    std::cerr << "[GPU " << device_id << "] ignoring YERBAS_CUDA_ACTIVE_BATCH="
                              << active_batch_env << " | valid range=1.." << batch_capacity << '\n';
                }
            } catch (...) {
                std::cerr << "[GPU " << device_id << "] ignoring invalid YERBAS_CUDA_ACTIVE_BATCH="
                          << active_batch_env << '\n';
            }
        }
    }

    ~Impl()
    {
        cudaSetDevice(device_id);
        if (d_candidate_count) cudaFree(d_candidate_count);
        if (d_candidates) cudaFree(d_candidates);
        if (d_cn_contexts) cudaFree(d_cn_contexts);
        if (d_cn_scratchpads) cudaFree(d_cn_scratchpads);
        if (d_states) cudaFree(d_states);
        if (d_nonces) cudaFree(d_nonces);
        if (d_job) cudaFree(d_job);
        if (stream) cudaStreamDestroy(stream);
    }
};

BatchEngine::BatchEngine(int device_id, std::size_t batch_size, unsigned int fallback_threads)
    : impl_(std::make_unique<Impl>(device_id, batch_size, fallback_threads)) {}
BatchEngine::~BatchEngine() = default;
BatchEngine::BatchEngine(BatchEngine&&) noexcept = default;
BatchEngine& BatchEngine::operator=(BatchEngine&&) noexcept = default;

void BatchEngine::upload_job(const JobDescriptor& job)
{
    impl_->host_job = job;
    DeviceJob device_job{};
    std::memcpy(device_job.header, job.header.data(), job.header.size());
    std::memcpy(device_job.target_le, job.target_le.data(), job.target_le.size());
    std::memcpy(device_job.stages, job.stages.data(), job.stages.size());
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemcpyAsync(impl_->d_job, &device_job, sizeof(device_job), cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync job failed");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize job failed");
    impl_->job_loaded = true;
}

std::vector<Candidate> BatchEngine::scan(std::uint32_t start_nonce)
{
    if (!impl_->job_loaded) throw std::runtime_error("CUDA batch scan requested before upload_job");
    if (!hash_pipeline_ready()) throw std::runtime_error("CUDA GhostRider scan requested before full native CUDA coverage is available");
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemsetAsync(impl_->d_candidate_count, 0, sizeof(unsigned int), impl_->stream), "cudaMemsetAsync candidate counter failed");
    constexpr int core_threads = 256;
    const int core_blocks = static_cast<int>((impl_->batch_size + core_threads - 1) / core_threads);
    initialize_nonce_batch<<<core_blocks, core_threads, 0, impl_->stream>>>(start_nonce, impl_->d_nonces, impl_->batch_size);
    check_cuda(cudaGetLastError(), "initialize_nonce_batch launch failed");

    ++impl_->scan_count;
    const bool profile_this_scan = (impl_->scan_count % 4U) == 0U;
    std::array<cudaEvent_t, 19> stage_boundaries{};
    if (profile_this_scan) {
        for (auto& event : stage_boundaries)
            check_cuda(cudaEventCreateWithFlags(&event, cudaEventDefault), "cudaEventCreate rolling stage profile failed");
        check_cuda(cudaEventRecord(stage_boundaries[0], impl_->stream), "cudaEventRecord rolling stage profile start failed");
    }

    for (int stage_index = 0; stage_index < 18; ++stage_index) {
        const std::uint8_t stage = impl_->host_job.stages[static_cast<std::size_t>(stage_index)];
        const bool is_cn = (stage & ghostrider::kCryptoNightStageFlag) != 0;
        const std::uint8_t algorithm = static_cast<std::uint8_t>(stage & 0x7fU);
        if (is_cn) {
            if (algorithm >= cryptonight::kVariantConfigs.size()) throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
            launch_split_cryptonight(impl_->stream, impl_->d_states, impl_->batch_size, algorithm,
                                     impl_->d_cn_scratchpads, impl_->d_cn_contexts, impl_->cn_geometries[algorithm]);
        } else {
            if (!core::core512_implemented(algorithm)) throw std::runtime_error("GhostRider conventional stage has no native CUDA implementation");
            core512_stage<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_job, impl_->d_nonces,
                                                                           impl_->d_states, impl_->batch_size,
                                                                           stage_index, algorithm);
            check_cuda(cudaGetLastError(), "GhostRider CUDA core stage launch failed");
        }
        if (profile_this_scan)
            check_cuda(cudaEventRecord(stage_boundaries[static_cast<std::size_t>(stage_index) + 1U], impl_->stream),
                       "cudaEventRecord rolling stage profile boundary failed");
    }
    collect_candidates<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_nonces, impl_->d_states, impl_->batch_size,
                                                                        impl_->d_job, impl_->d_candidates, impl_->d_candidate_count);
    check_cuda(cudaGetLastError(), "collect_candidates launch failed");
    unsigned int count = 0;
    check_cuda(cudaMemcpyAsync(&count, impl_->d_candidate_count, sizeof(count), cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync candidate count failed");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize scan failed");

    if (profile_this_scan) {
        ++impl_->stage_profile_samples;
        std::array<float, 18> sample_ms{};
        float sample_total_ms = 0.0F;
        for (std::size_t i = 0; i < sample_ms.size(); ++i) {
            check_cuda(cudaEventElapsedTime(&sample_ms[i], stage_boundaries[i], stage_boundaries[i + 1U]),
                       "cudaEventElapsedTime rolling stage profile failed");
            sample_total_ms += sample_ms[i];
            const std::uint8_t encoded = impl_->host_job.stages[i];
            impl_->rolling_stage_ms[encoded] += static_cast<double>(sample_ms[i]);
            ++impl_->rolling_stage_count[encoded];
        }
        for (auto& event : stage_boundaries) cudaEventDestroy(event);

        if ((impl_->stage_profile_samples % 4U) == 0U) {
            std::uint64_t fingerprint = 1469598103934665603ULL;
            for (const std::uint8_t stage : impl_->host_job.stages) {
                fingerprint ^= stage;
                fingerprint *= 1099511628211ULL;
            }
            std::cout << std::fixed << std::setprecision(3)
                      << "[GPU " << impl_->device_id << "] GhostRider rolling stage profile"
                      << " | samples=" << impl_->stage_profile_samples
                      << " | fingerprint=" << std::hex << std::setw(16) << std::setfill('0') << fingerprint
                      << std::dec << std::setfill(' ')
                      << " | sampled-stage-total=" << sample_total_ms << " ms\n";
            for (std::size_t i = 0; i < sample_ms.size(); ++i) {
                const std::uint8_t encoded = impl_->host_job.stages[i];
                const bool is_cn = (encoded & ghostrider::kCryptoNightStageFlag) != 0;
                const std::uint8_t algorithm = static_cast<std::uint8_t>(encoded & 0x7fU);
                const double avg_ms = impl_->rolling_stage_count[encoded] == 0 ? 0.0 :
                    impl_->rolling_stage_ms[encoded] / static_cast<double>(impl_->rolling_stage_count[encoded]);
                const float pct = sample_total_ms > 0.0F ? (100.0F * sample_ms[i] / sample_total_ms) : 0.0F;
                std::cout << "[CUDA stage] " << std::setw(2) << i << " ";
                if (is_cn && algorithm < cryptonight::kVariantConfigs.size()) {
                    std::cout << cryptonight::kVariantConfigs[algorithm].name;
                } else {
                    static constexpr const char* kCoreNames[15] = {
                        "BLAKE-512", "BMW-512", "Groestl-512", "JH-512", "Keccak-512",
                        "Skein-512", "Luffa-512", "CubeHash-512", "Shavite-512", "SIMD-512",
                        "Echo-512", "Hamsi-512", "Fugue-512", "Shabal-512", "Whirlpool"
                    };
                    std::cout << (algorithm < 15 ? kCoreNames[algorithm] : "UNKNOWN");
                }
                std::cout << " | sample=" << sample_ms[i] << " ms"
                          << " | sample=" << std::setprecision(1) << pct << "%"
                          << std::setprecision(3) << " | rolling-avg=" << avg_ms << " ms\n";
            }
            std::cout << std::defaultfloat;
        }
    }

    count = std::min<unsigned int>(count, static_cast<unsigned int>(kMaxCandidates));
    if (count == 0) return {};
    std::vector<DeviceCandidate> raw(count);
    check_cuda(cudaMemcpy(raw.data(), impl_->d_candidates, count * sizeof(DeviceCandidate), cudaMemcpyDeviceToHost),
               "cudaMemcpy candidates failed");
    std::vector<Candidate> out;
    out.reserve(count);
    for (const auto& item : raw) {
        Candidate candidate;
        candidate.nonce = item.nonce;
        std::copy(std::begin(item.hash), std::end(item.hash), candidate.hash.begin());
        out.push_back(candidate);
    }
    return out;
}

