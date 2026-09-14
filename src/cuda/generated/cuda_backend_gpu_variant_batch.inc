// Per-CryptoNight-variant GPU batch calibration.
//
// The existing scratchpad-class calibration remains the conservative fallback
// and memory-policy anchor. This layer refines it into six independent preferred
// batches, one for each GhostRider CryptoNight variant. Normal startup only loads
// the sidecar cache (or falls back to the class batches); explicit GPU autotune is
// the only path that benchmarks. Production still uses one common nonce count for
// all 18 GhostRider stages: upload_job() takes the minimum preferred batch across
// the exact CN variants present in the rotation, then applies the existing byte
// budget/capacity guard.

namespace {

constexpr int kGpuVariantBatchRevision = 1;
constexpr std::size_t kGpuVariantBatchCount = 6;
constexpr double kGpuVariantBatchMinimumGain = 1.01;

struct GpuVariantBatchPolicy {
    std::array<std::size_t, kGpuVariantBatchCount> batches{};
    bool specialized{false};
    bool from_cache{false};
};

bool gpu_variant_batch_autotune_requested()
{
    const char* explicit_value = std::getenv("YERBAS_GPU_VARIANT_AUTOTUNE");
    if (explicit_value && *explicit_value)
        return std::string(explicit_value) != "0";
    return gpu_calibration_requested();
}

std::filesystem::path gpu_variant_batch_cache_path(int device_id,
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
    name << "gpu-variant-batch-rev" << kGpuVariantBatchRevision
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

std::array<std::size_t, kGpuVariantBatchCount> gpu_variant_batch_defaults(
    const std::array<std::size_t, kGpuBatchClassCount>& class_batches,
    std::size_t capacity,
    std::size_t scratchpad_budget_bytes,
    std::size_t quantum)
{
    std::array<std::size_t, kGpuVariantBatchCount> defaults{};
    for (std::size_t variant = 0; variant < defaults.size(); ++variant) {
        const auto& cfg = cryptonight::kVariantConfigs[variant];
        const std::size_t cls = gpu_batch_class_index(cfg.page_size);
        const std::size_t safe_max = gpu_align_batch(
            std::min(capacity,
                     scratchpad_budget_bytes / std::max<std::size_t>(1U, cfg.page_size)),
            capacity, quantum);
        std::size_t value = class_batches[cls];
        if (value == 0U || value > safe_max) value = safe_max;
        defaults[variant] = gpu_align_batch(value, capacity, quantum);
    }
    return defaults;
}

bool load_gpu_variant_batch_cache(
    int device_id,
    const cudaDeviceProp& props,
    std::size_t capacity,
    std::size_t scratchpad_budget_bytes,
    std::size_t quantum,
    const std::array<std::size_t, kGpuBatchClassCount>& class_batches,
    GpuVariantBatchPolicy& policy)
{
    const auto path = gpu_variant_batch_cache_path(device_id, props, capacity);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    std::size_t cached_capacity = 0;
    std::size_t cached_budget = 0;
    std::array<std::size_t, kGpuBatchClassCount> cached_classes{};
    std::array<std::size_t, kGpuVariantBatchCount> cached_variants{};

    if (!(in >> magic >> revision >> cached_capacity >> cached_budget)) return false;
    for (auto& value : cached_classes) if (!(in >> value)) return false;
    for (auto& value : cached_variants) if (!(in >> value)) return false;

    if (magic != "YERBAS_GPU_VARIANT_BATCH" ||
        revision != kGpuVariantBatchRevision ||
        cached_capacity != capacity ||
        cached_budget != scratchpad_budget_bytes ||
        cached_classes != class_batches)
        return false;

    for (std::size_t variant = 0; variant < cached_variants.size(); ++variant) {
        const auto& cfg = cryptonight::kVariantConfigs[variant];
        const std::size_t safe_max = gpu_align_batch(
            std::min(capacity,
                     scratchpad_budget_bytes / std::max<std::size_t>(1U, cfg.page_size)),
            capacity, quantum);
        const std::size_t value = cached_variants[variant];
        if (value == 0U || value > safe_max || value > capacity)
            return false;
        if (value >= quantum && (value % quantum) != 0U)
            return false;
    }

    policy.batches = cached_variants;
    policy.specialized = true;
    policy.from_cache = true;

    std::cout << "[GPU " << device_id << "] GPU CN-variant batch cache loaded | batches=";
    for (std::size_t variant = 0; variant < policy.batches.size(); ++variant) {
        if (variant) std::cout << '/';
        std::cout << cryptonight::kVariantConfigs[variant].name << ':' << policy.batches[variant];
    }
    std::cout << " | startup-benchmark=off\n";
    return true;
}

void save_gpu_variant_batch_cache(
    int device_id,
    const cudaDeviceProp& props,
    std::size_t capacity,
    std::size_t scratchpad_budget_bytes,
    const std::array<std::size_t, kGpuBatchClassCount>& class_batches,
    const std::array<std::size_t, kGpuVariantBatchCount>& variant_batches)
{
    try {
        const auto path = gpu_variant_batch_cache_path(device_id, props, capacity);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;

        out << "YERBAS_GPU_VARIANT_BATCH " << kGpuVariantBatchRevision
            << ' ' << capacity << ' ' << scratchpad_budget_bytes;
        for (const auto value : class_batches) out << ' ' << value;
        for (const auto value : variant_batches) out << ' ' << value;
        out << '\n';
        out.close();
        if (!out) return;

        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id
                  << "] GPU CN-variant batch cache write skipped: "
                  << e.what() << '\n';
    }
}

DeviceJob make_gpu_variant_calibration_job(std::uint8_t target_variant)
{
    DeviceJob job{};
    std::array<std::uint8_t, 80> header{};
    header[0] = 4;
    for (std::size_t i = 4; i < 76; ++i)
        header[i] = static_cast<std::uint8_t>(
            (i * (37U + static_cast<unsigned int>(target_variant) * 11U) +
             0x63U + static_cast<unsigned int>(target_variant) * 29U) & 0xffU);

    const auto schedule = ghostrider::stage_schedule_quiet(
        ghostrider::Work{header.data(), header.size()});
    std::copy(header.begin(), header.end(), std::begin(job.header));
    std::fill(std::begin(job.target_le), std::end(job.target_le), 0xffU);
    std::copy(schedule.begin(), schedule.end(), std::begin(job.stages));

    // Keep the conventional 15-stage ordering from a real GhostRider schedule,
    // but make all three CN slots use the target variant. This produces a stable
    // full-pipeline workload that isolates that variant's batch-pressure behavior
    // without inventing a different nonce count inside the 18-stage pipeline.
    for (auto& encoded : job.stages) {
        if ((encoded & ghostrider::kCryptoNightStageFlag) == 0) continue;
        encoded = static_cast<std::uint8_t>(
            ghostrider::kCryptoNightStageFlag | target_variant);
    }
    return job;
}

GpuVariantBatchPolicy tune_gpu_variant_batches(
    int device_id,
    const cudaDeviceProp& props,
    const std::array<std::size_t, kGpuBatchClassCount>& class_batches,
    std::size_t capacity,
    std::size_t scratchpad_budget_bytes,
    std::size_t quantum,
    const std::array<int, kCoreAlgorithmCount>& core_threads,
    cudaStream_t stream,
    DeviceJob* d_job,
    std::uint32_t* d_nonces,
    std::uint8_t* states,
    std::uint8_t* scratchpads,
    cryptonight::SplitContext* contexts)
{
    GpuVariantBatchPolicy policy{};
    policy.batches = gpu_variant_batch_defaults(
        class_batches, capacity, scratchpad_budget_bytes, quantum);

    const int warp = std::max(1, props.warpSize);
    int calibration_core_threads = kCoreDefaultThreads;
    for (const int value : core_threads) {
        if (value > 0) {
            calibration_core_threads = value;
            break;
        }
    }

    std::cout << "[GPU " << device_id << "] GPU CN-variant batch autotune starting"
              << " | variants=6 | candidates<=6/variant"
              << " | metric=synthetic-variant GhostRider H/s"
              << " | gate=+1.00%"
              << " | class-policy=anchor"
              << " | memory-policy=allocated-byte-budget\n";

    for (std::size_t variant = 0; variant < policy.batches.size(); ++variant) {
        const auto& cfg = cryptonight::kVariantConfigs[variant];
        const std::size_t cls = gpu_batch_class_index(cfg.page_size);
        const std::size_t safe_max = gpu_align_batch(
            std::min(capacity,
                     scratchpad_budget_bytes / std::max<std::size_t>(1U, cfg.page_size)),
            capacity, quantum);
        const std::size_t nominal = gpu_align_batch(
            std::min(policy.batches[variant], safe_max), capacity, quantum);

        std::array<std::size_t, 6> raw_candidates{{
            gpu_align_batch((nominal * 3U) / 4U, capacity, quantum),
            gpu_align_batch((nominal * 7U) / 8U, capacity, quantum),
            nominal,
            gpu_align_batch(std::min(safe_max, nominal + std::max<std::size_t>(quantum, nominal / 8U)),
                            capacity, quantum),
            gpu_align_batch(std::min(safe_max, nominal + std::max<std::size_t>(quantum, nominal / 2U)),
                            capacity, quantum),
            safe_max
        }};

        std::array<std::size_t, 6> candidates{};
        std::size_t candidate_count = 0;
        for (const std::size_t raw : raw_candidates) {
            const std::size_t value = std::max<std::size_t>(1U, std::min(raw, safe_max));
            bool duplicate = false;
            for (std::size_t i = 0; i < candidate_count; ++i)
                duplicate = duplicate || candidates[i] == value;
            if (!duplicate) candidates[candidate_count++] = value;
        }

        const DeviceJob job = make_gpu_variant_calibration_job(
            static_cast<std::uint8_t>(variant));
        GpuCalibrationShape shape{};
        shape.batch = nominal;
        shape.core_threads = calibration_core_threads;
        shape.cn_threads = warp;

        // One untimed nominal warm-up settles lazy CUDA/runtime/kernel state so
        // candidate ordering does not reward the second measurement merely for
        // avoiding first-use overhead.
        (void)run_gpu_calibration_once(stream, d_job, d_nonces, states,
                                       scratchpads, contexts, job, shape);

        double nominal_hps = 0.0;
        double best_hps = -1.0;
        std::size_t best_batch = nominal;

        std::cout << "[GPU " << device_id << "] GPU CN batch search"
                  << " | variant=" << cfg.name
                  << " | class=" << kGpuBatchClassNames[cls]
                  << " | scratchpad=" << (cfg.page_size / 1024U) << " KiB/hash"
                  << " | nominal=" << nominal
                  << " | safe-max=" << safe_max
                  << " | candidates=";
        for (std::size_t i = 0; i < candidate_count; ++i) {
            if (i) std::cout << ',';
            std::cout << candidates[i];
        }
        std::cout << '\n';

        for (std::size_t i = 0; i < candidate_count; ++i) {
            shape.batch = candidates[i];
            const double hps = gpu_calibration_hps(
                stream, d_job, d_nonces, states, scratchpads, contexts, job, shape);
            if (candidates[i] == nominal) nominal_hps = hps;
            if (hps > best_hps) {
                best_hps = hps;
                best_batch = candidates[i];
            }
            std::cout << std::fixed << std::setprecision(2)
                      << "[GPU " << device_id << "] GPU CN batch sample"
                      << " | variant=" << cfg.name
                      << " | batch=" << candidates[i]
                      << " | throughput=" << hps << " H/s"
                      << std::defaultfloat << '\n';
        }

        std::size_t selected = nominal;
        bool promoted = false;
        if (best_hps > 0.0 && nominal_hps > 0.0 && best_batch != nominal &&
            best_hps > nominal_hps * kGpuVariantBatchMinimumGain) {
            selected = best_batch;
            promoted = true;
        }
        policy.batches[variant] = selected;

        const double gain_pct = nominal_hps > 0.0
            ? (best_hps / nominal_hps - 1.0) * 100.0 : 0.0;
        std::cout << std::fixed << std::setprecision(2)
                  << "[GPU " << device_id << "] GPU CN batch selected"
                  << " | variant=" << cfg.name
                  << " | nominal=" << nominal
                  << " | best=" << best_batch
                  << " | best-gain=" << (gain_pct >= 0.0 ? "+" : "") << gain_pct << '%'
                  << " | selected=" << selected
                  << " | decision=" << (promoted ? "promote" : "keep-nominal")
                  << std::defaultfloat << '\n';
    }

    save_gpu_variant_batch_cache(device_id, props, capacity,
                                 scratchpad_budget_bytes, class_batches,
                                 policy.batches);
    policy.specialized = true;
    policy.from_cache = false;

    std::cout << "[GPU " << device_id << "] GPU CN-variant batch autotune selected | batches=";
    for (std::size_t variant = 0; variant < policy.batches.size(); ++variant) {
        if (variant) std::cout << '/';
        std::cout << cryptonight::kVariantConfigs[variant].name << ':' << policy.batches[variant];
    }
    std::cout << " | cache=gpu-variant-batch-v" << kGpuVariantBatchRevision << '\n';
    return policy;
}

GpuVariantBatchPolicy build_gpu_variant_batch_policy(
    int device_id,
    const std::array<std::size_t, kGpuBatchClassCount>& class_batches,
    std::size_t capacity,
    std::size_t scratchpad_budget_bytes,
    std::size_t quantum,
    const std::array<int, kCoreAlgorithmCount>& core_threads,
    cudaStream_t stream,
    DeviceJob* d_job,
    std::uint32_t* d_nonces,
    std::uint8_t* states,
    std::uint8_t* scratchpads,
    cryptonight::SplitContext* contexts)
{
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties variant batch policy failed");

    GpuVariantBatchPolicy policy{};
    policy.batches = gpu_variant_batch_defaults(
        class_batches, capacity, scratchpad_budget_bytes, quantum);

    if (gpu_variant_batch_autotune_requested()) {
        try {
            return tune_gpu_variant_batches(
                device_id, props, class_batches, capacity,
                scratchpad_budget_bytes, quantum, core_threads, stream,
                d_job, d_nonces, states, scratchpads, contexts);
        } catch (const std::exception& e) {
            cudaGetLastError();
            std::cerr << "[GPU " << device_id
                      << "] GPU CN-variant batch autotune failed; class fallback retained: "
                      << e.what() << '\n';
            return policy;
        }
    }

    if (load_gpu_variant_batch_cache(device_id, props, capacity,
                                     scratchpad_budget_bytes, quantum,
                                     class_batches, policy))
        return policy;

    std::cout << "[GPU " << device_id << "] GPU CN-variant batch policy"
              << " | source=class-fallback | batches=";
    for (std::size_t variant = 0; variant < policy.batches.size(); ++variant) {
        if (variant) std::cout << '/';
        std::cout << cryptonight::kVariantConfigs[variant].name << ':' << policy.batches[variant];
    }
    std::cout << " | startup-benchmark=off\n";
    return policy;
}

} // anonymous namespace
