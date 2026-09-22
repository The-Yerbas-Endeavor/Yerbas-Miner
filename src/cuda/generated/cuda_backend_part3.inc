#include "cuda/generated/cuda_backend_gpu_variant_batch.inc"
#include "cuda/generated/cuda_backend_gpu_variant_runtime.inc"

struct BatchEngine::Impl {
    int device_id{-1};
    std::size_t batch_size{0};
    std::size_t base_batch_size{0};
    std::size_t batch_capacity{0};
    std::size_t scratchpad_budget_bytes{0};
    std::size_t batch_quantum{1};
    std::array<std::size_t, kGpuBatchClassCount> class_batches{};
    std::array<std::size_t, kGpuVariantBatchCount> variant_batches{};
    bool variant_batch_specialized{false};
    bool variant_batch_from_cache{false};
    unsigned int fallback_threads{1};
    cudaStream_t stream{};
    DeviceJob* d_job{nullptr};
    std::uint32_t* d_nonces{nullptr};
    std::uint8_t* d_states{nullptr};
    std::uint8_t* d_cn_scratchpads{nullptr};
    cryptonight::SplitContext* d_cn_contexts{nullptr};
    DeviceCandidate* d_candidates{nullptr};
    unsigned int* d_candidate_count{nullptr};
    DeviceCandidate* h_candidates{nullptr};
    unsigned int* h_candidate_count{nullptr};
    JobDescriptor host_job{};
    std::vector<std::uint8_t> host_states;
    std::array<CnGeometry, 6> cn_geometries{};
    std::array<int, kCoreAlgorithmCount> core_threads{};
    std::uint64_t scan_count{0};
    std::uint64_t stage_profile_samples{0};
    std::array<double, 256> rolling_stage_ms{};
    std::array<std::uint64_t, 256> rolling_stage_count{};
    bool diagnostics_enabled{false};
    bool benchmark_raw_batch{false};
    bool benchmark_stage_local{false};
    std::size_t benchmark_requested_batch{0};
    std::size_t benchmark_cn_chunk{0};
    std::size_t stale_batch_cap{0};
    std::size_t stale_batch_cap_min_tuned{0};
    bool job_loaded{false};

    Impl(int id, std::size_t requested_size, unsigned int requested_fallback_threads)
        : device_id(id),
          fallback_threads(std::max(1u, requested_fallback_threads))
    {
        const char* raw_batch_env = std::getenv("YERBAS_CUDA_BENCH_RAW_BATCH");
        benchmark_raw_batch =
            raw_batch_env != nullptr && *raw_batch_env != '\0' &&
            std::string(raw_batch_env) != "0";

        const char* stage_local_env = std::getenv("YERBAS_CUDA_BENCH_STAGE_LOCAL");
        benchmark_stage_local =
            stage_local_env != nullptr && *stage_local_env != '\0' &&
            std::string(stage_local_env) != "0";

        benchmark_requested_batch = requested_size;
        benchmark_cn_chunk = requested_size;

        // Experimental production-only stale-work cap. A global cap can be
        // supplied with YERBAS_GPU_STALE_BATCH_CAP, and an individual device
        // can override it with YERBAS_GPU_STALE_BATCH_CAP_<device-id>.
        // Zero/unset means uncapped. Benchmark modes intentionally ignore it.
        auto parse_batch_cap = [](const char* value) -> std::size_t {
            if (value == nullptr || *value == '\0') return 0U;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (end == value || *end != '\0') return 0U;
            return static_cast<std::size_t>(parsed);
        };
        stale_batch_cap = parse_batch_cap(std::getenv("YERBAS_GPU_STALE_BATCH_CAP"));
        const std::string device_cap_name =
            std::string("YERBAS_GPU_STALE_BATCH_CAP_") + std::to_string(device_id);
        if (const char* device_value = std::getenv(device_cap_name.c_str());
            device_value != nullptr && *device_value != '\0')
            stale_batch_cap = parse_batch_cap(device_value);
        if (benchmark_stage_local) {
            const char* chunk_env = std::getenv("YERBAS_CUDA_BENCH_CN_CHUNK");
            if (chunk_env != nullptr && *chunk_env != '\0') {
                const unsigned long parsed = std::strtoul(chunk_env, nullptr, 10);
                if (parsed != 0UL)
                    benchmark_cn_chunk = static_cast<std::size_t>(parsed);
            } else {
                benchmark_cn_chunk = std::min<std::size_t>(3584U, requested_size);
            }
        }

        batch_size = (benchmark_raw_batch || benchmark_stage_local)
            ? requested_size
            : std::min(requested_size, cryptonight::kInitialMaxBatch);
        base_batch_size = batch_size;

        if (base_batch_size == 0) throw std::runtime_error("CUDA batch size must be greater than zero");

        const char* diagnostics_env = std::getenv("YERBAS_DIAGNOSTICS");
        diagnostics_enabled = diagnostics_env && *diagnostics_env && std::string(diagnostics_env) != "0";

        // Keep a fixed scratchpad byte budget based on the conservative heavy-CN
        // baseline. Lightweight nonce/state/context slots are allocated up to the
        // smallest CN footprint so per-class calibration can choose among all
        // memory-safe 2MiB/1MiB/512KiB/256KiB batch sizes without reallocating.
        constexpr std::size_t min_cn_scratchpad = 262144U;
        const std::size_t heavy_budget_batch = benchmark_stage_local
            ? std::min(benchmark_cn_chunk, requested_size)
            : base_batch_size;

        std::size_t scratchpad_budget_stride = kCnScratchpadStride;
        if (benchmark_raw_batch) {
            if (const char* value = std::getenv("YERBAS_CUDA_BENCH_SCRATCHPAD_STRIDE");
                value != nullptr && *value != '\0') {
                char* end = nullptr;
                const unsigned long long parsed = std::strtoull(value, &end, 10);
                if (end != value && *end == '\0' &&
                    parsed >= min_cn_scratchpad &&
                    parsed <= kCnScratchpadStride &&
                    (parsed % min_cn_scratchpad) == 0U)
                    scratchpad_budget_stride = static_cast<std::size_t>(parsed);
            }
        }

        scratchpad_budget_bytes = heavy_budget_batch * scratchpad_budget_stride;
        const std::size_t slot_limit = scratchpad_budget_bytes / min_cn_scratchpad;
        batch_capacity = benchmark_stage_local
            ? requested_size
            : std::max(base_batch_size, std::min(requested_size, slot_limit));

        check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties adaptive batch failed");
        const std::size_t warp = static_cast<std::size_t>(std::max(1, props.warpSize));
        const std::size_t sms = static_cast<std::size_t>(std::max(1, props.multiProcessorCount));
        batch_quantum = std::max<std::size_t>(warp, warp * sms);

        host_states.resize(batch_capacity * kStateBytes);
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)), "cudaMalloc job failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces), batch_capacity * sizeof(std::uint32_t)), "cudaMalloc nonce batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_states), batch_capacity * kStateBytes), "cudaMalloc 512-bit state batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_scratchpads), scratchpad_budget_bytes), "cudaMalloc CryptoNight scratchpads failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_contexts), batch_capacity * sizeof(cryptonight::SplitContext)), "cudaMalloc CryptoNight split contexts failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_candidates), kMaxCandidates * sizeof(DeviceCandidate)), "cudaMalloc candidate buffer failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_candidate_count), sizeof(unsigned int)), "cudaMalloc candidate counter failed");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_candidates), kMaxCandidates * sizeof(DeviceCandidate)), "cudaMallocHost candidate buffer failed");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_candidate_count), sizeof(unsigned int)), "cudaMallocHost candidate counter failed");

        const auto policy = build_gpu_tune_policy(device_id, base_batch_size, batch_capacity, stream,
                                                  d_states, d_cn_scratchpads, d_cn_contexts);
        batch_size = policy.batch_size;
        base_batch_size = policy.class_batches[0] == 0U ? policy.batch_size : policy.class_batches[0];
        class_batches = policy.class_batches;
        if (class_batches[0] == 0U) {
            for (std::size_t cls = 0; cls < kGpuBatchClassCount; ++cls)
                class_batches[cls] = gpu_memory_derived_class_batch(
                    base_batch_size, batch_capacity, kGpuBatchClassPageBytes[cls], batch_quantum);
        }
        batch_size = class_batches[0];
        cn_geometries = policy.cn_geometries;
        core_threads = policy.core_threads;

        const auto variant_policy = build_gpu_variant_batch_policy(
            device_id, class_batches, batch_capacity, scratchpad_budget_bytes,
            batch_quantum, core_threads, stream, d_job, d_nonces, d_states,
            d_cn_scratchpads, d_cn_contexts);
        variant_batches = variant_policy.batches;
        variant_batch_specialized = variant_policy.specialized;
        variant_batch_from_cache = variant_policy.from_cache;

        if (benchmark_stage_local) {
            std::cout << "[GPU " << device_id << "] benchmark stage-local batching"
                      << " | outer=" << requested_size
                      << " | heavy-cn-chunk=" << benchmark_cn_chunk
                      << " | scratchpad-budget="
                      << (scratchpad_budget_bytes / (1024U * 1024U)) << " MiB"
                      << " | production-policy=bypassed\n";
        }

        std::cout << "[GPU " << device_id << "] adaptive batch capacity"
                  << " | calibrated=" << class_batches[0] << '/' << class_batches[1]
                  << '/' << class_batches[2] << '/' << class_batches[3]
                  << " | classes=2MiB/1MiB/512KiB/256KiB"
                  << " | max=" << batch_capacity
                  << " | scratchpad-budget=" << (scratchpad_budget_bytes / (1024U * 1024U)) << " MiB"
                  << " | policy=rotation-aware"
                  << " | variant-policy="
                  << (variant_batch_specialized
                          ? (variant_batch_from_cache ? "cache" : "tuned")
                          : "class-fallback")
                  << '\n';
    }

    void launch_cn_stage(std::uint8_t algorithm)
    {
        if (!benchmark_stage_local) {
            launch_split_cryptonight(
                stream, d_states, batch_size, algorithm,
                d_cn_scratchpads, d_cn_contexts, cn_geometries[algorithm]);
            return;
        }

        const auto& cfg = cryptonight::kVariantConfigs[algorithm];
        const std::size_t max_chunk = std::max<std::size_t>(
            1U, scratchpad_budget_bytes /
                std::max<std::size_t>(1U, cfg.page_size));
        const std::size_t chunk = std::min(batch_size, max_chunk);

        std::array<bool, kGpuVariantBatchCount> active_variant{};
        active_variant[algorithm] = true;

        std::cout << "[GPU " << device_id << "] stage-local CN"
                  << " | variant=" << cfg.name
                  << " | outer=" << batch_size
                  << " | chunk=" << chunk
                  << " | chunks=" << ((batch_size + chunk - 1U) / chunk)
                  << '\n';

        for (std::size_t offset = 0; offset < batch_size; offset += chunk) {
            const std::size_t count = std::min(chunk, batch_size - offset);
            gpu_variant_activate_runtime_batch(
                device_id, count, active_variant);
            launch_split_cryptonight(
                stream,
                d_states + offset * kStateBytes,
                count,
                algorithm,
                d_cn_scratchpads,
                d_cn_contexts + offset,
                cn_geometries[algorithm]);
        }
    }

    ~Impl()
    {
        cudaSetDevice(device_id);
        if (h_candidate_count) cudaFreeHost(h_candidate_count);
        if (h_candidates) cudaFreeHost(h_candidates);
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

    std::size_t max_page_size = 0;
    std::array<bool, kGpuVariantBatchCount> used_variants{};
    for (const std::uint8_t encoded : job.stages) {
        if ((encoded & ghostrider::kCryptoNightStageFlag) == 0) continue;
        const std::uint8_t variant = static_cast<std::uint8_t>(encoded & 0x7fU);
        if (variant >= cryptonight::kVariantConfigs.size()) continue;
        used_variants[variant] = true;
        max_page_size = std::max<std::size_t>(max_page_size,
                                              cryptonight::kVariantConfigs[variant].page_size);
    }
    if (max_page_size == 0) max_page_size = kCnScratchpadStride;

    const std::size_t class_index = gpu_batch_class_index(max_page_size);
    std::size_t adaptive_batch = impl_->class_batches[class_index];
    bool using_variant_policy = impl_->variant_batch_specialized;

    if (using_variant_policy) {
        adaptive_batch = impl_->batch_capacity;
        bool found_variant = false;
        for (std::size_t variant = 0; variant < used_variants.size(); ++variant) {
            if (!used_variants[variant]) continue;
            const std::size_t preferred = impl_->variant_batches[variant];
            if (preferred == 0U || preferred > impl_->batch_capacity) {
                using_variant_policy = false;
                break;
            }
            adaptive_batch = std::min(adaptive_batch, preferred);
            found_variant = true;
        }
        if (!found_variant) using_variant_policy = false;
    }

    if (!using_variant_policy) {
        adaptive_batch = impl_->class_batches[class_index];
        if (adaptive_batch == 0U || adaptive_batch > impl_->batch_capacity) {
            adaptive_batch = gpu_memory_derived_class_batch(
                impl_->base_batch_size, impl_->batch_capacity,
                max_page_size, impl_->batch_quantum);
        }
    }

    // Final byte-budget guard. A corrupted/stale cache can never select more CN
    // scratchpad bytes than were allocated for this engine. This also keeps a
    // specialized per-variant preference safe when several variants share one
    // GhostRider rotation: every one of the 18 stages still uses this same count.
    const std::size_t memory_safe_batch = std::max<std::size_t>(1U,
        std::min(impl_->batch_capacity, impl_->scratchpad_budget_bytes / max_page_size));

    if (impl_->benchmark_stage_local) {
        adaptive_batch = std::min(
            impl_->benchmark_requested_batch, impl_->batch_capacity);
    } else {
        adaptive_batch = std::min(adaptive_batch, memory_safe_batch);
        if (impl_->benchmark_raw_batch) {
            adaptive_batch = std::min(
                impl_->benchmark_requested_batch, memory_safe_batch);
        }
    }

    if (!impl_->benchmark_raw_batch && !impl_->benchmark_stage_local &&
        impl_->stale_batch_cap > 0U &&
        (impl_->stale_batch_cap_min_tuned == 0U ||
         adaptive_batch >= impl_->stale_batch_cap_min_tuned)) {
        std::size_t cap = std::min(
            impl_->stale_batch_cap,
            std::min(impl_->batch_capacity, memory_safe_batch));
        if (cap >= impl_->batch_quantum)
            cap = (cap / impl_->batch_quantum) * impl_->batch_quantum;
        cap = std::max<std::size_t>(1U, cap);

        if (adaptive_batch > cap) {
            std::cout << "[GPU " << impl_->device_id << "] stale-aware batch cap"
                      << " | tuned=" << adaptive_batch
                      << " -> capped=" << cap
                      << " | requested-cap=" << impl_->stale_batch_cap
                      << " | minimum-tuned=" << impl_->stale_batch_cap_min_tuned
                      << " | quantum=" << impl_->batch_quantum << '\n';
            adaptive_batch = cap;
        }
    }

    if (!impl_->benchmark_raw_batch && !impl_->benchmark_stage_local &&
        adaptive_batch >= impl_->batch_quantum)
        adaptive_batch =
            (adaptive_batch / impl_->batch_quantum) * impl_->batch_quantum;
    adaptive_batch = std::max<std::size_t>(1U, adaptive_batch);

    if (impl_->benchmark_stage_local) {
        std::cout << "[GPU " << impl_->device_id
                  << "] benchmark stage-local outer batch"
                  << " | requested=" << impl_->benchmark_requested_batch
                  << " | active=" << adaptive_batch
                  << " | heavy-cn-safe=" << memory_safe_batch
                  << " | production-policy=bypassed\n";
    } else if (impl_->benchmark_raw_batch) {
        std::cout << "[GPU " << impl_->device_id
                  << "] benchmark raw batch override"
                  << " | requested=" << impl_->benchmark_requested_batch
                  << " | active=" << adaptive_batch
                  << " | memory-safe=" << memory_safe_batch
                  << " | production-policy=bypassed\n";
    }

    // Every CN selector/cache is keyed by batch on disk. Mirror that contract in
    // live memory as rotations move between Fast/Lite/heavier preferred counts.
    // Partial explicit-retune progress is preserved and resumed when a batch
    // returns, rather than being discarded or reused at the wrong count.
    if (!impl_->benchmark_stage_local)
        gpu_variant_activate_runtime_batch(
            impl_->device_id, adaptive_batch, used_variants);

    if (adaptive_batch != impl_->batch_size) {
        std::cout << "[GPU " << impl_->device_id << "] rotation-adaptive batch"
                  << " | class=" << kGpuBatchClassNames[class_index]
                  << " | scratchpad=" << (max_page_size / 1024U) << " KiB/hash"
                  << " | batch=" << impl_->batch_size << " -> " << adaptive_batch
                  << " | policy=" << (using_variant_policy ? "variant-min" : "class")
                  << " | class-calibrated=" << impl_->class_batches[class_index];
        if (using_variant_policy) {
            std::cout << " | variants=";
            bool first = true;
            for (std::size_t variant = 0; variant < used_variants.size(); ++variant) {
                if (!used_variants[variant]) continue;
                if (!first) std::cout << '/';
                first = false;
                std::cout << cryptonight::kVariantConfigs[variant].name
                          << ':' << impl_->variant_batches[variant];
            }
        }
        std::cout << " | max=" << impl_->batch_capacity << '\n';
        impl_->batch_size = adaptive_batch;
    }

    DeviceJob device_job{};
    std::memcpy(device_job.header, job.header.data(), job.header.size());
    std::memcpy(device_job.target_le, job.target_le.data(), job.target_le.size());
    std::memcpy(device_job.stages, job.stages.data(), job.stages.size());
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemcpyAsync(impl_->d_job, &device_job, sizeof(device_job), cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync job failed");
    // upload_gpu_job() drains any active scan before mutating the engine. The
    // descriptor copy and every subsequent scan launch use this same stream, so
    // CUDA stream ordering guarantees the job copy completes before kernels read it.
    impl_->job_loaded = true;
}

std::vector<Candidate> BatchEngine::scan(std::uint32_t start_nonce)
{
    if (!impl_->job_loaded) throw std::runtime_error("CUDA batch scan requested before upload_job");
    if (!hash_pipeline_ready()) throw std::runtime_error("CUDA GhostRider scan requested before full native CUDA coverage is available");

    // Persistent GPU workers repeatedly call scan() from the same host thread.
    // Bind that thread to its CUDA device once instead of paying a runtime
    // cudaSetDevice call on every batch. This remains generic for any GPU count
    // or model and still rebinds correctly if a thread is ever reused.
    static thread_local int bound_device = -1;
    if (bound_device != impl_->device_id) {
        check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
        bound_device = impl_->device_id;
    }

    check_cuda(cudaMemsetAsync(impl_->d_candidate_count, 0, sizeof(unsigned int), impl_->stream), "cudaMemsetAsync candidate counter failed");
    constexpr int utility_threads = 256;
    const int utility_blocks = static_cast<int>((impl_->batch_size + utility_threads - 1) / utility_threads);
    initialize_nonce_batch<<<utility_blocks, utility_threads, 0, impl_->stream>>>(start_nonce, impl_->d_nonces, impl_->batch_size);
    check_cuda(cudaGetLastError(), "initialize_nonce_batch launch failed");

    ++impl_->scan_count;
    const bool profile_this_scan = impl_->diagnostics_enabled && (impl_->scan_count % 4U) == 0U;
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
            impl_->launch_cn_stage(algorithm);
        } else {
            if (!core::core512_implemented(algorithm)) throw std::runtime_error("GhostRider conventional stage has no native CUDA implementation");
            int stage_threads = kCoreDefaultThreads;
            if (algorithm < impl_->core_threads.size())
                stage_threads = impl_->core_threads[algorithm];
            const int stage_blocks = static_cast<int>((impl_->batch_size + static_cast<std::size_t>(stage_threads) - 1U) /
                                                      static_cast<std::size_t>(stage_threads));
            core512_stage<<<stage_blocks, stage_threads, 0, impl_->stream>>>(impl_->d_job, impl_->d_nonces,
                                                                             impl_->d_states, impl_->batch_size,
                                                                             stage_index, algorithm);
            check_cuda(cudaGetLastError(), "GhostRider CUDA core stage launch failed");
        }
        if (profile_this_scan)
            check_cuda(cudaEventRecord(stage_boundaries[static_cast<std::size_t>(stage_index) + 1U], impl_->stream),
                       "cudaEventRecord rolling stage profile boundary failed");
    }
    collect_candidates<<<utility_blocks, utility_threads, 0, impl_->stream>>>(impl_->d_nonces, impl_->d_states, impl_->batch_size,
                                                                               impl_->d_job, impl_->d_candidates, impl_->d_candidate_count);
    check_cuda(cudaGetLastError(), "collect_candidates launch failed");

    // Queue both tiny result readbacks before the one scan synchronization. The
    // pinned host buffers make the D2H copies truly asynchronous, and copying the
    // fixed 64-entry candidate buffer (~2.3 KiB) avoids a second synchronous
    // cudaMemcpy on candidate-bearing scans. This is deliberately lightweight:
    // it does not duplicate the memory-heavy CryptoNight scratchpad/state arena.
    check_cuda(cudaMemcpyAsync(impl_->h_candidate_count, impl_->d_candidate_count, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync candidate count failed");
    check_cuda(cudaMemcpyAsync(impl_->h_candidates, impl_->d_candidates,
                               kMaxCandidates * sizeof(DeviceCandidate),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync candidate buffer failed");
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

        std::uint64_t fingerprint = 14695981039346656037ULL;
        for (const std::uint8_t stage : impl_->host_job.stages) {
            fingerprint ^= stage;
            fingerprint *= 1099511628211ULL;
        }

        const double gr_hps = sample_total_ms > 0.0F
            ? (static_cast<double>(impl_->batch_size) * 1000.0 / static_cast<double>(sample_total_ms))
            : 0.0;
        std::ostringstream cn_names;
        bool first_cn = true;
        for (const std::uint8_t encoded : impl_->host_job.stages) {
            if ((encoded & ghostrider::kCryptoNightStageFlag) == 0) continue;
            const std::uint8_t algorithm = static_cast<std::uint8_t>(encoded & 0x7fU);
            if (algorithm >= cryptonight::kVariantConfigs.size()) continue;
            if (!first_cn) cn_names << '/';
            cn_names << cryptonight::kVariantConfigs[algorithm].name;
            first_cn = false;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA GR perf] GPU " << impl_->device_id
                  << " | fingerprint=" << std::hex << std::setw(16) << std::setfill('0') << fingerprint
                  << std::dec << std::setfill(' ')
                  << " | CN=" << cn_names.str()
                  << " | batch=" << impl_->batch_size
                  << " | stages=" << sample_total_ms << " ms"
                  << " | throughput=" << gr_hps << " H/s"
                  << std::defaultfloat << '\n';

        if ((impl_->stage_profile_samples % 4U) == 0U) {
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
                    std::cout << (algorithm < kCoreAlgorithmNames.size() ?
                                  kCoreAlgorithmNames[algorithm] : "UNKNOWN");
                }
                std::cout << " | sample=" << sample_ms[i] << " ms"
                          << " | sample=" << std::setprecision(1) << pct << "%"
                          << std::setprecision(3) << " | rolling-avg=" << avg_ms << " ms\n";
            }
            std::cout << std::defaultfloat;
        }
    }

    const unsigned int count = std::min<unsigned int>(*impl_->h_candidate_count,
                                                        static_cast<unsigned int>(kMaxCandidates));
    if (count == 0) return {};
    std::vector<Candidate> out;
    out.reserve(count);
    for (unsigned int i = 0; i < count; ++i) {
        const auto& item = impl_->h_candidates[i];
        Candidate candidate;
        candidate.nonce = item.nonce;
        std::copy(std::begin(item.hash), std::end(item.hash), candidate.hash.begin());
        out.push_back(candidate);
    }
    return out;
}
