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

constexpr int kGpuVariantBatchRevision = 2;
constexpr std::size_t kGpuVariantBatchCount = 6;
constexpr double kGpuVariantBatchMinimumGain = 1.01;
constexpr int kGpuVariantBatchConfirmPasses = 2;

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

std::size_t gpu_variant_batch_cache_capacity_hint(const std::filesystem::path& path)
{
    const std::string name = path.filename().string();
    const std::size_t cap_pos = name.find("-cap");
    if (cap_pos == std::string::npos) return 0U;
    const std::size_t begin = cap_pos + 4U;
    const std::size_t end = name.find("-drv", begin);
    if (end == std::string::npos || end <= begin) return 0U;
    try {
        return static_cast<std::size_t>(std::stoull(name.substr(begin, end - begin)));
    } catch (...) {
        return 0U;
    }
}

std::vector<std::filesystem::path> gpu_variant_batch_cache_candidates(
    int device_id,
    const cudaDeviceProp& props,
    std::size_t capacity)
{
    std::vector<std::filesystem::path> candidates;
    const auto exact = gpu_variant_batch_cache_path(device_id, props, capacity);
    if (exact.empty()) return candidates;
    candidates.push_back(exact);

    const std::string exact_name = exact.filename().string();
    const std::size_t cap_pos = exact_name.find("-cap");
    const std::size_t drv_pos = exact_name.find("-drv", cap_pos);
    if (cap_pos == std::string::npos || drv_pos == std::string::npos) return candidates;

    const std::string prefix = exact_name.substr(0, cap_pos + 4U);
    const std::string suffix = exact_name.substr(drv_pos);

    std::error_code ec;
    std::filesystem::directory_iterator it(exact.parent_path(), ec);
    const std::filesystem::directory_iterator end;
    while (!ec && it != end) {
        const auto candidate = it->path();
        const std::string name = candidate.filename().string();
        const bool prefix_match =
            name.size() >= prefix.size() &&
            name.compare(0, prefix.size(), prefix) == 0;
        const bool suffix_match =
            name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        if (prefix_match && suffix_match && candidate != exact)
            candidates.push_back(candidate);
        it.increment(ec);
    }

    if (candidates.size() > 2U) {
        std::stable_sort(candidates.begin() + 1, candidates.end(),
            [capacity](const auto& a, const auto& b) {
                const std::size_t ca = gpu_variant_batch_cache_capacity_hint(a);
                const std::size_t cb = gpu_variant_batch_cache_capacity_hint(b);
                const std::size_t da = ca > capacity ? ca - capacity : capacity - ca;
                const std::size_t db = cb > capacity ? cb - capacity : capacity - cb;
                return da < db;
            });
    }
    return candidates;
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
    const auto paths = gpu_variant_batch_cache_candidates(device_id, props, capacity);
    if (paths.empty()) return false;

    for (const auto& path : paths) {
        std::ifstream in(path);
        if (!in) continue;

        std::string magic;
        int revision = 0;
        std::size_t cached_capacity = 0;
        std::size_t cached_budget = 0;
        std::array<std::size_t, kGpuBatchClassCount> cached_classes{};
        std::array<std::size_t, kGpuVariantBatchCount> cached_variants{};

        if (!(in >> magic >> revision >> cached_capacity >> cached_budget)) continue;
        bool parsed = true;
        for (auto& value : cached_classes)
            parsed = parsed && static_cast<bool>(in >> value);
        for (auto& value : cached_variants)
            parsed = parsed && static_cast<bool>(in >> value);
        if (!parsed) continue;

        if (magic != "YERBAS_GPU_VARIANT_BATCH" ||
            revision != kGpuVariantBatchRevision ||
            cached_capacity == 0U || cached_budget == 0U ||
            cached_classes != class_batches)
            continue;

        bool safe = true;
        for (std::size_t variant = 0; variant < cached_variants.size(); ++variant) {
            const auto& cfg = cryptonight::kVariantConfigs[variant];
            const std::size_t safe_max = gpu_align_batch(
                std::min(capacity,
                         scratchpad_budget_bytes / std::max<std::size_t>(1U, cfg.page_size)),
                capacity, quantum);
            const std::size_t value = cached_variants[variant];
            if (value == 0U || value > safe_max || value > capacity ||
                (value >= quantum && (value % quantum) != 0U)) {
                safe = false;
                break;
            }
        }
        if (!safe) continue;

        policy.batches = cached_variants;
        policy.specialized = true;
        policy.from_cache = true;

        std::cout << "[GPU " << device_id << "] GPU CN-variant batch cache loaded | batches=";
        for (std::size_t variant = 0; variant < policy.batches.size(); ++variant) {
            if (variant) std::cout << '/';
            std::cout << cryptonight::kVariantConfigs[variant].name << ':' << policy.batches[variant];
        }
        if (cached_capacity != capacity || cached_budget != scratchpad_budget_bytes)
            std::cout << " | capacity-reuse=" << cached_capacity << "->" << capacity;
        std::cout << " | startup-benchmark=off\n";
        return true;
    }
    return false;
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
              << " | confirmation=median-of-3 winner-vs-nominal"
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

        double decision_nominal_hps = nominal_hps;
        double decision_best_hps = best_hps;
        bool confirmation_run = false;

        // A full multi-pass sweep would make explicit tuning much too expensive.
        // Instead, only a provisional promotion pays for two extra measurements
        // of the winner and nominal. Alternate their order across the two passes,
        // then compare median-of-three values including the original sweep sample.
        // This removes most candidate-order/temperature noise while preserving a
        // bounded startup cost.
        if (best_batch != nominal && best_hps > 0.0 && nominal_hps > 0.0 &&
            best_hps > nominal_hps * kGpuVariantBatchMinimumGain) {
            confirmation_run = true;
            std::array<double, 3> nominal_samples{{nominal_hps, 0.0, 0.0}};
            std::array<double, 3> best_samples{{best_hps, 0.0, 0.0}};

            for (int pass = 0; pass < kGpuVariantBatchConfirmPasses; ++pass) {
                const bool nominal_first = (pass % 2) == 0;
                for (int slot = 0; slot < 2; ++slot) {
                    const bool measure_nominal = (slot == 0) == nominal_first;
                    const std::size_t batch = measure_nominal ? nominal : best_batch;
                    shape.batch = batch;
                    const double hps = gpu_calibration_hps(
                        stream, d_job, d_nonces, states, scratchpads, contexts, job, shape);
                    if (measure_nominal)
                        nominal_samples[static_cast<std::size_t>(pass) + 1U] = hps;
                    else
                        best_samples[static_cast<std::size_t>(pass) + 1U] = hps;

                    std::cout << std::fixed << std::setprecision(2)
                              << "[GPU " << device_id << "] GPU CN batch confirm"
                              << " | variant=" << cfg.name
                              << " | pass=" << (pass + 1) << '/' << kGpuVariantBatchConfirmPasses
                              << " | batch=" << batch
                              << " | role=" << (measure_nominal ? "nominal" : "winner")
                              << " | throughput=" << hps << " H/s"
                              << std::defaultfloat << '\n';
                }
            }

            std::sort(nominal_samples.begin(), nominal_samples.end());
            std::sort(best_samples.begin(), best_samples.end());
            decision_nominal_hps = nominal_samples[1];
            decision_best_hps = best_samples[1];

            std::cout << std::fixed << std::setprecision(2)
                      << "[GPU " << device_id << "] GPU CN batch confirmation"
                      << " | variant=" << cfg.name
                      << " | nominal=" << nominal << '@' << decision_nominal_hps << " H/s"
                      << " | winner=" << best_batch << '@' << decision_best_hps << " H/s"
                      << " | metric=median-of-3"
                      << std::defaultfloat << '\n';
        }

        std::size_t selected = nominal;
        bool promoted = false;
        if (decision_best_hps > 0.0 && decision_nominal_hps > 0.0 &&
            best_batch != nominal &&
            decision_best_hps > decision_nominal_hps * kGpuVariantBatchMinimumGain) {
            selected = best_batch;
            promoted = true;
        }
        policy.batches[variant] = selected;

        const double gain_pct = decision_nominal_hps > 0.0
            ? (decision_best_hps / decision_nominal_hps - 1.0) * 100.0 : 0.0;
        std::cout << std::fixed << std::setprecision(2)
                  << "[GPU " << device_id << "] GPU CN batch selected"
                  << " | variant=" << cfg.name
                  << " | nominal=" << nominal
                  << " | best=" << best_batch
                  << " | best-gain=" << (gain_pct >= 0.0 ? "+" : "") << gain_pct << '%'
                  << " | selected=" << selected
                  << " | decision=" << (promoted ? "promote" : "keep-nominal")
                  << " | evidence=" << (confirmation_run ? "median-of-3" : "sweep")
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
