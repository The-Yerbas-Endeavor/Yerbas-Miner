// Per-device launch-width tuner for the 15 conventional GhostRider core stages.
// Unlike the retired CN-Fast micro-experiments, this targets most of the
// 18-stage GhostRider pipeline. Results are cached by GPU/driver/runtime/batch.

namespace {

constexpr int kCoreThreadCacheRevision = 2;
constexpr int kCoreAlgorithmCount = 15;
constexpr int kCoreDefaultThreads = 256;
constexpr float kCoreThreadRequiredSpeedup = 0.98F; // require >=2% over 256

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
    return threads > 0 && threads <= props.maxThreadsPerBlock &&
           threads <= kernel_max_threads &&
           (threads == 64 || threads == 128 || threads == 256 || threads == 512);
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
        // Cache failure is non-fatal; default 256-thread launches remain safe.
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

std::array<int, kCoreAlgorithmCount> autotune_core_threads(int device_id,
                                                            cudaStream_t stream,
                                                            std::uint8_t* states,
                                                            std::size_t batch_size)
{
    std::array<int, kCoreAlgorithmCount> result{};
    result.fill(kCoreDefaultThreads);
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties core launch tuner failed");

    cudaFuncAttributes attrs{};
    check_cuda(cudaFuncGetAttributes(&attrs, benchmark_core512_stage),
               "cudaFuncGetAttributes core launch tuner failed");
    const int kernel_max_threads = std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock);
    if (kernel_max_threads < kCoreDefaultThreads) {
        std::cout << "[GPU " << device_id << "] core launch autotune skipped"
                  << " | kernel_max_threads=" << kernel_max_threads
                  << " | production=256\n";
        return result;
    }

    if (load_core_thread_cache(device_id, props, batch_size, kernel_max_threads, result)) return result;

    const std::size_t sample_count = std::min<std::size_t>(batch_size, 1024U);
    constexpr std::array<int, 4> candidates{{64, 128, 256, 512}};
    std::cout << "[GPU " << device_id << "] core launch autotune"
              << " | kernel_max_threads=" << kernel_max_threads
              << " | regs=" << attrs.numRegs
              << " | local=" << attrs.localSizeBytes
              << " | shared=" << attrs.sharedSizeBytes << '\n';

    for (int algorithm = 0; algorithm < kCoreAlgorithmCount; ++algorithm) {
        if (!core::core512_implemented(static_cast<std::uint8_t>(algorithm))) continue;
        const float baseline_ms = benchmark_core_threads(stream,
            static_cast<std::uint8_t>(algorithm), states, sample_count, kCoreDefaultThreads);
        float best_ms = baseline_ms;
        int best_threads = kCoreDefaultThreads;
        for (const int threads : candidates) {
            if (!valid_core_threads(threads, props, kernel_max_threads) || threads == kCoreDefaultThreads) continue;
            try {
                const float a = benchmark_core_threads(stream, static_cast<std::uint8_t>(algorithm),
                                                       states, sample_count, threads);
                const float b = benchmark_core_threads(stream, static_cast<std::uint8_t>(algorithm),
                                                       states, sample_count, threads);
                const float ms = std::min(a, b);
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
        const bool qualified = best_threads != kCoreDefaultThreads &&
                               best_ms < baseline_ms * kCoreThreadRequiredSpeedup;
        result[static_cast<std::size_t>(algorithm)] = qualified ? best_threads : kCoreDefaultThreads;
        const float gain = best_ms > 0.0F ? (baseline_ms / best_ms - 1.0F) * 100.0F : 0.0F;
        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA core tune] GPU " << device_id
                  << " | " << kCoreAlgorithmNames[static_cast<std::size_t>(algorithm)]
                  << " | baseline=256:" << baseline_ms << " ms"
                  << " | best=" << best_threads << ':' << best_ms << " ms"
                  << " | gain=" << gain << "%"
                  << " | selected=" << result[static_cast<std::size_t>(algorithm)]
                  << std::defaultfloat << '\n';
    }
    save_core_thread_cache(device_id, props, batch_size, result);
    return result;
}

} // anonymous namespace
