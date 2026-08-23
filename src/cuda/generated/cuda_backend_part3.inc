struct BatchEngine::Impl {
    int device_id{-1};
    std::size_t batch_size{0};
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
    bool job_loaded{false};

    Impl(int id, std::size_t requested_size, unsigned int requested_fallback_threads)
        : device_id(id), batch_size(std::min(requested_size, cryptonight::kInitialMaxBatch)),
          fallback_threads(std::max(1u, requested_fallback_threads)), host_states(batch_size * kStateBytes)
    {
        if (batch_size == 0) throw std::runtime_error("CUDA batch size must be greater than zero");
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)), "cudaMalloc job failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces), batch_size * sizeof(std::uint32_t)), "cudaMalloc nonce batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_states), batch_size * kStateBytes), "cudaMalloc 512-bit state batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_scratchpads), batch_size * kCnScratchpadStride), "cudaMalloc CryptoNight scratchpads failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_contexts), batch_size * sizeof(cryptonight::SplitContext)), "cudaMalloc CryptoNight split contexts failed");
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
    }
    collect_candidates<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_nonces, impl_->d_states, impl_->batch_size,
                                                                        impl_->d_job, impl_->d_candidates, impl_->d_candidate_count);
    check_cuda(cudaGetLastError(), "collect_candidates launch failed");
    unsigned int count = 0;
    check_cuda(cudaMemcpyAsync(&count, impl_->d_candidate_count, sizeof(count), cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync candidate count failed");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize scan failed");
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

