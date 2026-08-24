// Production active-batch autotuner.
//
// Chooses the CUDA mining batch before geometry/backend tuning. The search is
// bounded by adaptive_batch_capacity(), uses aligned candidate sizes, measures
// all six CryptoNight variants with the established legacy split path, compares
// throughput rather than raw latency, and caches the winner per GPU/driver/runtime.
// YERBAS_CUDA_ACTIVE_BATCH remains an explicit override.

namespace {

constexpr int kCudaBatchTuneRevision = 1;
constexpr std::size_t kCudaBatchAlignment = 128U;

void throw_if_cuda_autotune_cancelled()
{
    if (::yerbas::stop_requested())
        throw std::runtime_error("CUDA autotune cancelled by user");
}

std::filesystem::path cuda_batch_cache_path(int device_id,
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
    key << "rev" << kCudaBatchTuneRevision
        << "-dev" << device_id
        << '-' << sanitize_cache_component(props.name)
        << "-cc" << props.major << props.minor
        << "-sm" << props.multiProcessorCount
        << "-warp" << props.warpSize
        << "-cap" << capacity
        << "-drv" << driver_version
        << "-rt" << runtime_version;
    return cn_cache_directory() / ("cuda-active-batch-" + key.str() + ".txt");
}

std::size_t align_cuda_batch(std::size_t value, std::size_t capacity)
{
    if (capacity < kCudaBatchAlignment) return capacity;
    value = std::max(kCudaBatchAlignment, std::min(value, capacity));
    value = (value / kCudaBatchAlignment) * kCudaBatchAlignment;
    if (value == 0) value = kCudaBatchAlignment;
    return std::min(value, capacity);
}

std::vector<std::size_t> cuda_batch_candidates(std::size_t initial,
                                                std::size_t capacity)
{
    std::vector<std::size_t> out;
    auto add = [&](std::size_t value) {
        value = align_cuda_batch(value, capacity);
        if (value == 0) return;
        if (std::find(out.begin(), out.end(), value) == out.end()) out.push_back(value);
    };

    add(initial / 2U);
    add((initial * 3U) / 4U);
    add(initial);
    add((capacity * 3U) / 4U);
    add((capacity * 7U) / 8U);
    add(capacity);
    std::sort(out.begin(), out.end());
    return out;
}

bool load_cuda_batch_cache(int device_id,
                           const cudaDeviceProp& props,
                           std::size_t capacity,
                           std::size_t& selected)
{
    const char* retune_all = std::getenv("YERBAS_CUDA_RETUNE");
    const char* retune_batch = std::getenv("YERBAS_CUDA_BATCH_RETUNE");
    if ((retune_all && *retune_all && std::string(retune_all) != "0") ||
        (retune_batch && *retune_batch && std::string(retune_batch) != "0"))
        return false;

    const auto path = cuda_batch_cache_path(device_id, props, capacity);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    std::string magic;
    int revision = 0;
    std::size_t stored_capacity = 0;
    std::size_t stored_batch = 0;
    if (!(in >> magic >> revision >> stored_capacity >> stored_batch) ||
        magic != "YERBAS_CUDA_BATCH" || revision != kCudaBatchTuneRevision ||
        stored_capacity != capacity || stored_batch == 0 || stored_batch > capacity)
        return false;
    selected = stored_batch;
    return true;
}

void save_cuda_batch_cache(int device_id,
                           const cudaDeviceProp& props,
                           std::size_t capacity,
                           std::size_t selected)
{
    try {
        const auto path = cuda_batch_cache_path(device_id, props, capacity);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CUDA_BATCH " << kCudaBatchTuneRevision << ' '
            << capacity << ' ' << selected << '\n';
        out.close();
        if (!out) return;
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (...) {
    }
}

template <std::uint8_t VariantIndex>
void launch_cuda_batch_cn_variant(cudaStream_t stream,
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

void launch_cuda_batch_cn_variant(std::uint8_t variant,
                                  cudaStream_t stream,
                                  std::uint8_t* states,
                                  std::size_t count,
                                  std::uint8_t* scratchpads,
                                  cryptonight::SplitContext* contexts,
                                  const CnGeometry& geometry)
{
    switch (variant) {
    case 0: launch_cuda_batch_cn_variant<0>(stream, states, count, scratchpads, contexts, geometry); break;
    case 1: launch_cuda_batch_cn_variant<1>(stream, states, count, scratchpads, contexts, geometry); break;
    case 2: launch_cuda_batch_cn_variant<2>(stream, states, count, scratchpads, contexts, geometry); break;
    case 3: launch_cuda_batch_cn_variant<3>(stream, states, count, scratchpads, contexts, geometry); break;
    case 4: launch_cuda_batch_cn_variant<4>(stream, states, count, scratchpads, contexts, geometry); break;
    case 5: launch_cuda_batch_cn_variant<5>(stream, states, count, scratchpads, contexts, geometry); break;
    default: throw std::runtime_error("CUDA batch autotune received invalid CryptoNight variant");
    }
}

float benchmark_cuda_batch_candidate(int device_id,
                                     cudaStream_t stream,
                                     std::uint8_t* states,
                                     std::size_t count,
                                     std::uint8_t* scratchpads,
                                     cryptonight::SplitContext* contexts)
{
    (void)device_id;
    CnGeometry geometry{};
    float total_ms = 0.0F;
    constexpr int repeats = 2;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        throw_if_cuda_autotune_cancelled();
        for (std::uint8_t variant = 0; variant < 6; ++variant) {
            throw_if_cuda_autotune_cancelled();
            check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31 + variant),
                                       count * kStateBytes, stream),
                       "CUDA batch autotune state reset failed");
            total_ms += time_cuda_phase(stream, [&]() {
                launch_cuda_batch_cn_variant(variant, stream, states, count,
                                             scratchpads, contexts, geometry);
            });
            throw_if_cuda_autotune_cancelled();
            check_cuda(cudaGetLastError(), "CUDA batch autotune CryptoNight launch failed");
        }
    }
    if (!(total_ms > 0.0F)) return 0.0F;
    return static_cast<float>((static_cast<double>(count) * 6.0 * repeats * 1000.0) /
                              static_cast<double>(total_ms));
}

std::size_t autotune_cuda_active_batch(int device_id,
                                       std::size_t initial_batch,
                                       std::size_t capacity,
                                       cudaStream_t stream,
                                       std::uint8_t* states,
                                       std::uint8_t* scratchpads,
                                       cryptonight::SplitContext* contexts)
{
    throw_if_cuda_autotune_cancelled();
    if (capacity == 0) return initial_batch;

    const char* override_env = std::getenv("YERBAS_CUDA_ACTIVE_BATCH");
    if (override_env && *override_env) {
        try {
            const std::size_t requested = static_cast<std::size_t>(std::stoull(override_env));
            if (requested > 0 && requested <= capacity) {
                std::cout << "[GPU " << device_id << "] CUDA active batch override | active="
                          << requested << " | capacity=" << capacity << '\n';
                return requested;
            }
            std::cerr << "[GPU " << device_id << "] ignoring YERBAS_CUDA_ACTIVE_BATCH="
                      << override_env << " | valid range=1.." << capacity << '\n';
        } catch (...) {
            std::cerr << "[GPU " << device_id << "] ignoring invalid YERBAS_CUDA_ACTIVE_BATCH="
                      << override_env << '\n';
        }
    }

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties batch autotune failed");

    std::size_t cached = 0;
    if (load_cuda_batch_cache(device_id, props, capacity, cached)) {
        std::cout << "[GPU " << device_id << "] CUDA active batch | source=cache | active="
                  << cached << " | capacity=" << capacity << '\n';
        return cached;
    }

    const auto candidates = cuda_batch_candidates(initial_batch, capacity);
    std::size_t best_batch = align_cuda_batch(initial_batch, capacity);
    float best_hps = 0.0F;
    const bool diagnostics = [] {
        const char* value = std::getenv("YERBAS_DIAGNOSTICS");
        return value && *value && std::string(value) != "0";
    }();

    std::cout << "[GPU " << device_id << "] CUDA active-batch autotune starting | candidates="
              << candidates.size() << " | capacity=" << capacity << '\n';

    for (const std::size_t candidate : candidates) {
        throw_if_cuda_autotune_cancelled();
        float hps = 0.0F;
        try {
            hps = benchmark_cuda_batch_candidate(device_id, stream, states, candidate,
                                                 scratchpads, contexts);
        } catch (const std::exception& e) {
            cudaGetLastError();
            if (::yerbas::stop_requested()) throw;
            if (diagnostics)
                std::cerr << "[GPU " << device_id << "] CUDA batch candidate " << candidate
                          << " failed: " << e.what() << '\n';
            continue;
        }
        if (diagnostics) {
            std::cout << std::fixed << std::setprecision(2)
                      << "[CUDA batch test] GPU " << device_id
                      << " | batch=" << candidate << " | CN throughput=" << hps << " H/s\n"
                      << std::defaultfloat;
        }
        if (hps > best_hps) {
            best_hps = hps;
            best_batch = candidate;
        }
    }

    throw_if_cuda_autotune_cancelled();
    save_cuda_batch_cache(device_id, props, capacity, best_batch);
    std::cout << std::fixed << std::setprecision(2)
              << "[GPU " << device_id << "] CUDA active batch selected | active=" << best_batch
              << " | capacity=" << capacity << " | CN throughput=" << best_hps << " H/s | cached=yes\n"
              << std::defaultfloat;
    return best_batch;
}

} // namespace
