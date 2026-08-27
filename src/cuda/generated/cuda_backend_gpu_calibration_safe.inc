// Safety wrapper for scratchpad-class GPU calibration.
//
// The class calibrator must never assume that every theoretical scratchpad
// class is represented by a valid GhostRider three-CN schedule.  Before any
// measurement, derive the actual largest CN scratchpad in the representative
// job and clamp the batch to the fixed scratchpad byte budget.  If a requested
// class has no matching representative schedule, inherit the next-heavier
// class result instead of benchmarking a fabricated fallback workload.

namespace {

std::size_t gpu_calibration_job_max_page(const DeviceJob& job)
{
    std::size_t max_page = 0U;
    for (const std::uint8_t encoded : job.stages) {
        if ((encoded & ghostrider::kCryptoNightStageFlag) == 0) continue;
        const std::uint8_t variant = static_cast<std::uint8_t>(encoded & 0x7fU);
        if (variant >= cryptonight::kVariantConfigs.size()) continue;
        max_page = std::max<std::size_t>(max_page,
                                         cryptonight::kVariantConfigs[variant].page_size);
    }
    return max_page == 0U ? kGpuBatchClassPageBytes[0] : max_page;
}

GpuTunePolicy run_gpu_calibration_safe(int device_id,
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
    const std::size_t quantum = std::max<std::size_t>(
        static_cast<std::size_t>(warp),
        static_cast<std::size_t>(warp) *
            static_cast<std::size_t>(std::max(1, props.multiProcessorCount)));
    const int core_threads = faststart_core_threads(props, kernel_max);
    const auto jobs = make_gpu_calibration_jobs_by_class();

    GpuCalibrationShape baseline{};
    baseline.batch = std::max<std::size_t>(1U, std::min(requested_batch, capacity));
    baseline.core_threads = core_threads;
    baseline.cn_threads = warp;

    const std::size_t scratchpad_budget = baseline.batch * kGpuBatchClassPageBytes[0];

    DeviceJob* d_job = nullptr;
    std::uint32_t* d_nonces = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)),
               "cudaMalloc GPU calibration job failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces),
                              capacity * sizeof(std::uint32_t)),
                   "cudaMalloc GPU calibration nonces failed");

        std::array<std::size_t, kGpuBatchClassCount> class_batches{};
        std::array<double, kGpuBatchClassCount> class_hps{};
        constexpr unsigned int total_measurements =
            static_cast<unsigned int>(kGpuBatchClassCount * 2U);
        unsigned int progress = 0U;

        std::cout << "[GPU " << device_id << "] GPU scratchpad-class autotune starting"
                  << " | classes=4 | measurements<=" << total_measurements
                  << " | metric=whole-GhostRider H/s"
                  << " | thread-width-search=off"
                  << " | safety=actual-scratchpad-clamp\n";

        for (std::size_t cls = 0; cls < kGpuBatchClassCount; ++cls) {
            const std::size_t actual_page = gpu_calibration_job_max_page(jobs[cls]);
            const std::size_t actual_cls = gpu_batch_class_index(actual_page);

            // A missing class in make_gpu_calibration_jobs_by_class() currently
            // contains a heavier fallback job. Never run that heavier job with
            // the lighter class batch. Inherit the previous heavier result.
            if (actual_cls != cls) {
                ++progress;
                const std::size_t inherited = cls == 0U
                    ? baseline.batch
                    : class_batches[cls - 1U];
                class_batches[cls] = inherited;
                class_hps[cls] = cls == 0U ? 0.0 : class_hps[cls - 1U];
                std::cout << "[GPU " << device_id << "] GPU autotune "
                          << progress << '/' << total_measurements
                          << " | class=" << kGpuBatchClassNames[cls]
                          << " | representative unavailable"
                          << " | actual-class=" << kGpuBatchClassNames[actual_cls]
                          << " | CN=" << gpu_calibration_cn_summary(jobs[cls])
                          << " | inherit-safe-batch=" << inherited << '\n';
                ++progress;
                std::cout << "[GPU " << device_id << "] GPU autotune "
                          << progress << '/' << total_measurements
                          << " | class=" << kGpuBatchClassNames[cls]
                          << " | measurement skipped safely\n";
                continue;
            }

            const std::size_t max_safe_batch = gpu_align_batch(
                scratchpad_budget / std::max<std::size_t>(1U, actual_page),
                capacity, quantum);
            std::size_t nominal = gpu_memory_derived_class_batch(
                baseline.batch, capacity, kGpuBatchClassPageBytes[cls], quantum);
            nominal = std::min(nominal, max_safe_batch);
            nominal = gpu_align_batch(nominal, capacity, quantum);
            const std::size_t reduced = gpu_align_batch(
                std::min(max_safe_batch, (nominal * 7U) / 8U), capacity, quantum);

            GpuCalibrationShape best = baseline;
            best.batch = nominal;

            ++progress;
            std::cout << "[GPU " << device_id << "] GPU autotune "
                      << progress << '/' << total_measurements
                      << " | class=" << kGpuBatchClassNames[cls]
                      << " | CN=" << gpu_calibration_cn_summary(jobs[cls])
                      << " | actual-scratchpad=" << (actual_page / 1024U) << " KiB/hash"
                      << " | safe-max=" << max_safe_batch
                      << " | batch=" << nominal << " | testing..." << std::flush;
            double best_hps = gpu_calibration_hps(
                stream, d_job, d_nonces, states, scratchpads, contexts,
                jobs[cls], best);
            std::cout << std::fixed << std::setprecision(2)
                      << " done | " << best_hps << " H/s | best="
                      << best_hps << " H/s" << std::defaultfloat << '\n';

            ++progress;
            if (reduced == nominal) {
                std::cout << "[GPU " << device_id << "] GPU autotune "
                          << progress << '/' << total_measurements
                          << " | class=" << kGpuBatchClassNames[cls]
                          << " | reduced candidate duplicates nominal | skipped\n";
            } else {
                GpuCalibrationShape candidate = baseline;
                candidate.batch = reduced;
                std::cout << "[GPU " << device_id << "] GPU autotune "
                          << progress << '/' << total_measurements
                          << " | class=" << kGpuBatchClassNames[cls]
                          << " | batch=" << reduced << " | testing..." << std::flush;
                const double hps = gpu_calibration_hps(
                    stream, d_job, d_nonces, states, scratchpads, contexts,
                    jobs[cls], candidate);
                const double gain_pct = best_hps > 0.0
                    ? (hps / best_hps - 1.0) * 100.0 : 0.0;
                const bool wins = hps > best_hps * kGpuCalibrationMinimumGain;
                std::cout << std::fixed << std::setprecision(2)
                          << " done | " << hps << " H/s"
                          << " | gain=" << (gain_pct >= 0.0 ? "+" : "")
                          << gain_pct << '%'
                          << " | gate=+1.00% | "
                          << (wins ? "NEW BEST" : "keep nominal")
                          << std::defaultfloat << '\n';
                if (wins) {
                    best = candidate;
                    best_hps = hps;
                }
            }

            class_batches[cls] = best.batch;
            class_hps[cls] = best_hps;
            std::cout << std::fixed << std::setprecision(2)
                      << "[GPU " << device_id << "] GPU class selected"
                      << " | class=" << kGpuBatchClassNames[cls]
                      << " | batch=" << best.batch
                      << " | throughput=" << best_hps << " H/s"
                      << std::defaultfloat << '\n';
        }

        // Monotonic safety: lighter classes may never receive a smaller batch
        // than a heavier class, but an unavailable class is allowed to inherit.
        for (std::size_t cls = 1; cls < kGpuBatchClassCount; ++cls)
            class_batches[cls] = std::max(class_batches[cls], class_batches[cls - 1U]);

        GpuCalibrationShape selected = baseline;
        selected.batch = class_batches[0];
        std::cout << "[GPU " << device_id << "] GPU autotune validation"
                  << " | exact parity..." << std::flush;
        const bool parity = gpu_calibration_parity(
            stream, d_job, d_nonces, states, scratchpads, contexts,
            jobs[0], baseline, selected);
        std::cout << " done | parity=" << (parity ? "PASS" : "FAIL") << '\n';

        if (!parity) {
            std::cerr << "[GPU " << device_id
                      << "] GPU autotune parity FAIL; conservative class defaults retained\n";
            class_batches[0] = baseline.batch;
            for (std::size_t cls = 1; cls < kGpuBatchClassCount; ++cls) {
                const std::size_t actual_page = gpu_calibration_job_max_page(jobs[cls]);
                if (gpu_batch_class_index(actual_page) != cls) {
                    class_batches[cls] = class_batches[cls - 1U];
                } else {
                    class_batches[cls] = gpu_memory_derived_class_batch(
                        baseline.batch, capacity, kGpuBatchClassPageBytes[cls], quantum);
                }
            }
            selected.batch = class_batches[0];
        }

        std::cout << "[GPU " << device_id
                  << "] GPU autotune saving calibration cache..." << std::flush;
        save_gpu_calibration_cache(device_id, props, capacity, selected,
                                   class_batches, class_hps[0]);
        std::cout << " done\n";

        GpuTunePolicy policy{};
        apply_calibration_shape(policy, selected);
        policy.class_batches = class_batches;
        policy.batch_size = class_batches[0];
        policy.throughput_hps = class_hps[0];
        policy.from_cache = false;

        std::cout << "[GPU " << device_id << "] GPU autotune selected"
                  << " | batches=" << class_batches[0] << '/'
                  << class_batches[1] << '/' << class_batches[2] << '/'
                  << class_batches[3]
                  << " | classes=2MiB/1MiB/512KiB/256KiB"
                  << " | core=" << core_threads
                  << " | CN=" << warp << '/' << warp << '/' << warp
                  << " | parity=" << (parity ? "PASS" : "defaults")
                  << " | cache=gpu-calibration-v" << kGpuCalibrationRevision << '\n';

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
    const int kernel_max = std::max(
        1, std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock));

    if (gpu_calibration_requested()) {
        return run_gpu_calibration_safe(device_id, props, kernel_max,
                                        requested_batch, capacity, stream,
                                        states, scratchpads, contexts);
    }

    GpuTunePolicy cached{};
    if (load_gpu_calibration_cache(device_id, props, capacity, kernel_max, cached))
        return cached;

    const int core_threads = faststart_core_threads(props, kernel_max);
    const int warp = std::max(1, props.warpSize);
    const std::size_t quantum = std::max<std::size_t>(
        static_cast<std::size_t>(warp),
        static_cast<std::size_t>(warp) *
            static_cast<std::size_t>(std::max(1, props.multiProcessorCount)));

    GpuTunePolicy policy{};
    policy.batch_size = std::max<std::size_t>(1U,
                                              std::min(requested_batch, capacity));
    for (std::size_t cls = 0; cls < kGpuBatchClassCount; ++cls)
        policy.class_batches[cls] = gpu_memory_derived_class_batch(
            policy.batch_size, capacity, kGpuBatchClassPageBytes[cls], quantum);
    policy.batch_size = policy.class_batches[0];
    policy.core_threads.fill(core_threads);
    for (auto& geometry : policy.cn_geometries) {
        geometry = CnGeometry{};
        geometry.setup_threads = warp;
        geometry.loop_threads = warp;
        geometry.final_threads = warp;
        geometry.aes_backend = 0;
    }

    std::cout << "[GPU " << device_id << "] production GPU fast-start"
              << " | batches=" << policy.class_batches[0] << '/'
              << policy.class_batches[1] << '/' << policy.class_batches[2]
              << '/' << policy.class_batches[3]
              << " | classes=2MiB/1MiB/512KiB/256KiB"
              << " | core=" << core_threads
              << " | CN=" << warp << '/' << warp << '/' << warp
              << " | source=runtime-safe | startup-benchmark=off\n";
    return policy;
}
