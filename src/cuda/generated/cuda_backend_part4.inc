std::vector<Candidate> BatchEngine::scan_profiled(std::uint32_t start_nonce, BatchProfile& profile)
{
    if (!impl_->job_loaded) throw std::runtime_error("CUDA profiled batch scan requested before upload_job");
    if (!hash_pipeline_ready()) throw std::runtime_error("CUDA GhostRider scan requested before full native CUDA coverage is available");
    profile = BatchProfile{};
    profile.hashes = impl_->batch_size;
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemsetAsync(impl_->d_candidate_count, 0, sizeof(unsigned int), impl_->stream), "cudaMemsetAsync candidate counter failed");
    constexpr int core_threads = 256;
    const int core_blocks = static_cast<int>((impl_->batch_size + core_threads - 1) / core_threads);
    cudaEvent_t total_start{}, total_stop{}, section_start{}, section_stop{};
    check_cuda(cudaEventCreate(&total_start), "cudaEventCreate total_start failed");
    check_cuda(cudaEventCreate(&total_stop), "cudaEventCreate total_stop failed");
    check_cuda(cudaEventCreate(&section_start), "cudaEventCreate section_start failed");
    check_cuda(cudaEventCreate(&section_stop), "cudaEventCreate section_stop failed");
    auto elapsed = [&](float& out) {
        check_cuda(cudaEventRecord(section_stop, impl_->stream), "cudaEventRecord section stop failed");
        check_cuda(cudaEventSynchronize(section_stop), "cudaEventSynchronize section stop failed");
        check_cuda(cudaEventElapsedTime(&out, section_start, section_stop), "cudaEventElapsedTime failed");
    };
    try {
        check_cuda(cudaEventRecord(total_start, impl_->stream), "cudaEventRecord total start failed");
        check_cuda(cudaEventRecord(section_start, impl_->stream), "cudaEventRecord nonce start failed");
        initialize_nonce_batch<<<core_blocks, core_threads, 0, impl_->stream>>>(start_nonce, impl_->d_nonces, impl_->batch_size);
        check_cuda(cudaGetLastError(), "initialize_nonce_batch launch failed");
        elapsed(profile.nonce_init_ms);
        for (int stage_index = 0; stage_index < 18; ++stage_index) {
            const std::uint8_t stage = impl_->host_job.stages[static_cast<std::size_t>(stage_index)];
            const bool is_cn = (stage & ghostrider::kCryptoNightStageFlag) != 0;
            const std::uint8_t algorithm = static_cast<std::uint8_t>(stage & 0x7fU);
            check_cuda(cudaEventRecord(section_start, impl_->stream), "cudaEventRecord stage start failed");
            if (is_cn) {
                if (algorithm >= cryptonight::kVariantConfigs.size()) throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
                impl_->launch_cn_stage(algorithm);
            } else {
                if (!core::core512_implemented(algorithm)) throw std::runtime_error("GhostRider conventional stage has no native CUDA implementation");
                core512_stage<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_job, impl_->d_nonces,
                                                                               impl_->d_states, impl_->batch_size,
                                                                               stage_index, algorithm);
                check_cuda(cudaGetLastError(), "GhostRider CUDA core stage launch failed");
            }
            float stage_ms = 0.0F;
            elapsed(stage_ms);
            profile.stages[static_cast<std::size_t>(stage_index)] = StageTiming{stage_index, stage, stage_ms};
            profile.stage_total_ms += stage_ms;
        }
        check_cuda(cudaEventRecord(section_start, impl_->stream), "cudaEventRecord candidate start failed");
        collect_candidates<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_nonces, impl_->d_states, impl_->batch_size,
                                                                            impl_->d_job, impl_->d_candidates, impl_->d_candidate_count);
        check_cuda(cudaGetLastError(), "collect_candidates launch failed");
        elapsed(profile.candidate_ms);
        check_cuda(cudaEventRecord(total_stop, impl_->stream), "cudaEventRecord total stop failed");
        check_cuda(cudaEventSynchronize(total_stop), "cudaEventSynchronize total stop failed");
        check_cuda(cudaEventElapsedTime(&profile.total_gpu_ms, total_start, total_stop), "cudaEventElapsedTime total failed");
        unsigned int count = 0;
        check_cuda(cudaMemcpyAsync(&count, impl_->d_candidate_count, sizeof(count), cudaMemcpyDeviceToHost, impl_->stream),
                   "cudaMemcpyAsync candidate count failed");
        check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize profiled scan failed");
        count = std::min<unsigned int>(count, static_cast<unsigned int>(kMaxCandidates));
        std::vector<Candidate> out;
        if (count != 0) {
            std::vector<DeviceCandidate> raw(count);
            check_cuda(cudaMemcpy(raw.data(), impl_->d_candidates, count * sizeof(DeviceCandidate), cudaMemcpyDeviceToHost),
                       "cudaMemcpy candidates failed");
            out.reserve(count);
            for (const auto& item : raw) {
                Candidate candidate;
                candidate.nonce = item.nonce;
                std::copy(std::begin(item.hash), std::end(item.hash), candidate.hash.begin());
                out.push_back(candidate);
            }
        }
        cudaEventDestroy(section_stop);
        cudaEventDestroy(section_start);
        cudaEventDestroy(total_stop);
        cudaEventDestroy(total_start);
        return out;
    } catch (...) {
        cudaEventDestroy(section_stop);
        cudaEventDestroy(section_start);
        cudaEventDestroy(total_stop);
        cudaEventDestroy(total_start);
        throw;
    }
}

void BatchEngine::set_active_batch_size(std::size_t batch_size)
{
    if (batch_size == 0 || batch_size > impl_->batch_capacity)
        throw std::out_of_range("CUDA active batch size exceeds allocated capacity");
    impl_->batch_size = batch_size;
}

std::size_t BatchEngine::try_grow_fast_lite_batch(std::size_t baseline_batch)
{
    if (!impl_->job_loaded || impl_->benchmark_raw_batch ||
        impl_->benchmark_stage_local || impl_->batch_size != baseline_batch)
        return impl_->batch_size;

    std::array<bool, kGpuVariantBatchCount> variants{};
    std::size_t max_page = 0U;
    for (const auto encoded : impl_->host_job.stages) {
        if ((encoded & ghostrider::kCryptoNightStageFlag) == 0U) continue;
        const auto variant = static_cast<std::size_t>(encoded & 0x7fU);
        if (variant >= variants.size()) return impl_->batch_size;
        variants[variant] = true;
        max_page = std::max(max_page,
            static_cast<std::size_t>(cryptonight::kVariantConfigs[variant].page_size));
    }
    if (!variants[2] || !variants[3] || max_page == 0U) return impl_->batch_size;

    const std::size_t safe_limit = std::min(impl_->batch_capacity,
        impl_->scratchpad_budget_bytes / max_page);
    std::size_t target = stratum::fast_lite_growth_target(baseline_batch, safe_limit);
    if (impl_->stale_batch_cap > 0U &&
        (impl_->stale_batch_cap_min_tuned == 0U ||
         target >= impl_->stale_batch_cap_min_tuned)) {
        target = stratum::fast_lite_growth_target(baseline_batch,
            std::min(safe_limit, impl_->stale_batch_cap));
    }
    if (target <= baseline_batch) return impl_->batch_size;

    // Switch batch-keyed selector/geometry state before the next launch.
    // This retains existing cache entries and does not persist a new preference.
    gpu_variant_activate_runtime_batch(impl_->device_id, target, variants);
    impl_->batch_size = target;
    impl_->fast_lite_grown_job = true;
    return target;
}

void BatchEngine::set_stale_batch_cap(std::size_t batch_size,
                                      std::size_t minimum_tuned_batch)
{
    impl_->stale_batch_cap = batch_size;
    impl_->stale_batch_cap_min_tuned = minimum_tuned_batch;
}

int BatchEngine::device_id() const noexcept { return impl_->device_id; }
std::size_t BatchEngine::batch_size() const noexcept { return impl_->batch_size; }
std::size_t BatchEngine::batch_capacity() const noexcept { return impl_->batch_capacity; }
bool BatchEngine::hash_pipeline_ready() const noexcept { return full_ghostrider_cuda_coverage(); }

Hash512 keccak512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length)
{
    return run_validation_kernel(device_id, input, length, true);
}

Hash512 cubehash512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length)
{
    return run_validation_kernel(device_id, input, length, false);
}

int device_count()
{
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return 0;
    }
    check_cuda(status, "cudaGetDeviceCount failed");
    return count;
}

std::vector<DeviceInfo> enumerate_devices()
{
    std::vector<DeviceInfo> devices;
    const int count = device_count();
    devices.reserve(static_cast<std::size_t>(count));
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties failed");
        devices.push_back(DeviceInfo{device, props.major, props.minor, props.totalGlobalMem,
                                     props.multiProcessorCount, props.warpSize});
    }
    return devices;
}

void print_devices()
{
    for (const auto& info : enumerate_devices()) {
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, info.id), "cudaGetDeviceProperties failed");
        const double memory_gib = static_cast<double>(info.total_memory) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "GPU " << info.id << ": " << props.name
                  << " | CC " << info.compute_major << '.' << info.compute_minor
                  << " | SMs " << info.multiprocessors
                  << " | warp " << info.warp_size
                  << " | " << memory_gib << " GiB\n";
    }
}

} // namespace yerbas::cuda

int yerbas_cuda_device_count() { return yerbas::cuda::device_count(); }
void yerbas_cuda_print_devices() { yerbas::cuda::print_devices(); }
