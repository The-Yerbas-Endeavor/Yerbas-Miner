// Generic per-device launch-width tuner for the 15 conventional GhostRider
// core stages. Candidate widths and the baseline are derived from CUDA-reported
// device/kernel limits; there are no GPU-model or architecture-specific rules.

namespace {

constexpr int kCoreThreadCacheRevision = 3;
constexpr int kCoreAlgorithmCount = 15;
constexpr int kCoreDefaultThreads = 256; // conservative fallback only
constexpr float kCoreThreadRequiredSpeedup = 0.98F; // require >=2% over baseline

static constexpr std::array<const char*, kCoreAlgorithmCount> kCoreAlgorithmNames{{
    "BLAKE-512", "BMW-512", "Groestl-512", "JH-512", "Keccak-512",
    "Skein-512", "Luffa-512", "CubeHash-512", "Shavite-512", "SIMD-512",
    "Echo-512", "Hamsi-512", "Fugue-512", "Shabal-512", "Whirlpool"
}};

__global__ void benchmark_core512_stage(std::uint8_t algorithm,
                                        std::uint8_t* states,
                                        std::size_t count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    std::uint8_t digest[64];
    std::uint8_t* state = states + index * kStateBytes;
    if (!core::dispatch_core512(algorithm, state, kStateBytes, digest)) return;
    #pragma unroll
    for (int i = 0; i < 64; ++i) state[i] = digest[i];
}

int aligned_down(int value, int alignment)
{
    if (alignment <= 0) return value;
    return (value / alignment) * alignment;
}

int safe_core_baseline_threads(const cudaDeviceProp& props,
                               const cudaFuncAttributes& attrs)
{
    const int warp = std::max(1, props.warpSize);
    const int limit = std::max(warp, std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock));
    const int preferred = std::min(kCoreDefaultThreads, limit);
    return std::max(warp, aligned_down(preferred, warp));
}

std::vector<int> core_thread_candidates(const cudaDeviceProp& props,
                                        const cudaFuncAttributes& attrs)
{
    const int warp = std::max(1, props.warpSize);
    const int limit = std::max(warp, std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock));
    std::vector<int> out;
    auto add = [&](int value) {
        value = aligned_down(std::min(value, limit), warp);
        if (value < warp) return;
        if (std::find(out.begin(), out.end(), value) == out.end()) out.push_back(value);
    };

    // Explore warp multiples from the hardware rather than a fixed NVIDIA model
    // table. Cap the matrix so startup remains practical on very large limits.
    add(warp);
    add(warp * 2);
    add(warp * 4);
    add(warp * 8);
    add(warp * 16);
    add(limit);
    add(safe_core_baseline_threads(props, attrs));
    std::sort(out.begin(), out.end());
    return out;
}

std::filesystem::path core_thread_cache_path(int device_id,
                                             const cudaDeviceProp& props,
                                             std::size_t batch_size)
{
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, batch_size,
                                         driver_version, runtime_version);
    return cn_cache_directory() / ("core-threads-rev" +
        std::to_string(kCoreThreadCacheRevision) + '-' + key + ".txt");
}

bool valid_core_threads(int threads, const cudaDeviceProp& props, int kernel_max_threads)
{
    const int warp = std::max(1, props.warpSize);
    return threads >= warp && threads <= props.maxThreadsPerBlock &&
           threads <= kernel_max_threads && (threads % warp) == 0;
}

bool load_core_thread_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size,
                            int kernel_max_threads,
                            std::array<int, kCoreAlgorithmCount>& result)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = core_thread_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;
    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision) || magic != "YERBAS_CORE_THREADS" ||
        revision != kCoreThreadCacheRevision) return false;
    std::array<int, kCoreAlgorithmCount> loaded{};
    for (int i = 0; i < kCoreAlgorithmCount; ++i) {
        int algorithm = -1;
        int threads = 0;
        if (!(in >> algorithm >> threads) || algorithm != i ||
            !valid_core_threads(threads, props, kernel_max_threads)) return false;
        loaded[static_cast<std::size_t>(i)] = threads;
    }
    result = loaded;
    std::cout << "[GPU " << device_id << "] core launch cache loaded | " << path.string() << '\n';
    return true;
}

void save_core_thread_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size,
                            const std::array<int, kCoreAlgorithmCount>& values)
{
    try {
        const auto path = core_thread_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CORE_THREADS " << kCoreThreadCacheRevision << '\n';
        for (int i = 0; i < kCoreAlgorithmCount; ++i)
            out << i << ' ' << values[static_cast<std::size_t>(i)] << '\n';
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
        // Cache failure is non-fatal; the runtime-derived baseline remains safe.
    }
}

float benchmark_core_threads(cudaStream_t stream,
                             std::uint8_t algorithm,
                             std::uint8_t* states,
                             std::size_t count,
                             int threads)
{
    check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + algorithm * 7U),
                               count * kStateBytes, stream),
               "core launch tuner state reset failed");
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    return time_cuda_phase(stream, [&]() {
        benchmark_core512_stage<<<blocks, threads, 0, stream>>>(algorithm, states, count);
        check_cuda(cudaGetLastError(), "core launch tuner kernel failed");
    });
}

float benchmark_core_threads_stable(cudaStream_t stream,
                                    std::uint8_t algorithm,
                                    std::uint8_t* states,
                                    std::size_t count,
                                    int threads)
{
    std::array<float, 3> samples{{
        benchmark_core_threads(stream, algorithm, states, count, threads),
        benchmark_core_threads(stream, algorithm, states, count, threads),
        benchmark_core_threads(stream, algorithm, states, count, threads)
    }};
    std::sort(samples.begin(), samples.end());
    return samples[1];
}

std::array<int, kCoreAlgorithmCount> autotune_core_threads(int device_id,
                                                            cudaStream_t stream,
                                                            std::uint8_t* states,
                                                            std::size_t batch_size)
{
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties core launch tuner failed");

    cudaFuncAttributes attrs{};
    check_cuda(cudaFuncGetAttributes(&attrs, benchmark_core512_stage),
               "cudaFuncGetAttributes core launch tuner failed");
    const int kernel_max_threads = std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock);
    const int baseline_threads = safe_core_baseline_threads(props, attrs);
    const auto candidates = core_thread_candidates(props, attrs);

    std::array<int, kCoreAlgorithmCount> result{};
    result.fill(baseline_threads);
    if (load_core_thread_cache(device_id, props, batch_size, kernel_max_threads, result)) return result;

    // Scale the sample with SM count and warp size, bounded by the active batch.
    const std::size_t occupancy_sample = static_cast<std::size_t>(std::max(1, props.multiProcessorCount)) *
                                         static_cast<std::size_t>(std::max(1, props.warpSize)) * 4U;
    const std::size_t sample_count = std::max<std::size_t>(1U,
        std::min(batch_size, std::max<std::size_t>(256U, occupancy_sample)));

    std::cout << "[GPU " << device_id << "] core launch autotune"
              << " | warp=" << props.warpSize
              << " | kernel_max_threads=" << kernel_max_threads
              << " | baseline=" << baseline_threads
              << " | candidates=" << candidates.size()
              << " | sample=" << sample_count
              << " | regs=" << attrs.numRegs
              << " | local=" << attrs.localSizeBytes
              << " | shared=" << attrs.sharedSizeBytes << '\n';

    for (int algorithm = 0; algorithm < kCoreAlgorithmCount; ++algorithm) {
        if (!core::core512_implemented(static_cast<std::uint8_t>(algorithm))) continue;
        const float baseline_ms = benchmark_core_threads_stable(stream,
            static_cast<std::uint8_t>(algorithm), states, sample_count, baseline_threads);
        float best_ms = baseline_ms;
        int best_threads = baseline_threads;
        for (const int threads : candidates) {
            if (!valid_core_threads(threads, props, kernel_max_threads) || threads == baseline_threads) continue;
            try {
                const float ms = benchmark_core_threads_stable(stream,
                    static_cast<std::uint8_t>(algorithm), states, sample_count, threads);
                if (ms < best_ms) {
                    best_ms = ms;
                    best_threads = threads;
                }
            } catch (const std::exception& e) {
                cudaGetLastError();
                std::cout << "[CUDA core tune] GPU " << device_id
                          << " | " << kCoreAlgorithmNames[static_cast<std::size_t>(algorithm)]
                          << " | threads=" << threads << " skipped: " << e.what() << '\n';
            }
        }
        const bool qualified = best_threads != baseline_threads &&
                               best_ms < baseline_ms * kCoreThreadRequiredSpeedup;
        result[static_cast<std::size_t>(algorithm)] = qualified ? best_threads : baseline_threads;
        const float gain = best_ms > 0.0F ? (baseline_ms / best_ms - 1.0F) * 100.0F : 0.0F;
        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA core tune] GPU " << device_id
                  << " | " << kCoreAlgorithmNames[static_cast<std::size_t>(algorithm)]
                  << " | baseline=" << baseline_threads << ':' << baseline_ms << " ms"
                  << " | best=" << best_threads << ':' << best_ms << " ms"
                  << " | gain=" << gain << "%"
                  << " | selected=" << result[static_cast<std::size_t>(algorithm)]
                  << std::defaultfloat << '\n';
    }
    save_core_thread_cache(device_id, props, batch_size, result);
    return result;
}

} // anonymous namespace
