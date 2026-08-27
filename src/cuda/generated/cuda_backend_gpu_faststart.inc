// Instant-start production GPU policy.
//
// Production mining must not block on synthetic benchmarking.  We derive one
// conservative legal policy from runtime CUDA limits, optionally reuse a
// compatible cached policy from the previous production tuner, and start mining
// immediately.  Longer-term optimization is driven by live GhostRider data.

constexpr int kCoreAlgorithmCount = 15;
constexpr int kCoreDefaultThreads = 256;
static constexpr std::array<const char*, kCoreAlgorithmCount> kCoreAlgorithmNames{{
    "BLAKE-512", "BMW-512", "Groestl-512", "JH-512", "Keccak-512",
    "Skein-512", "Luffa-512", "CubeHash-512", "Shavite-512", "SIMD-512",
    "Echo-512", "Hamsi-512", "Fugue-512", "Shabal-512", "Whirlpool"
}};

struct GpuTunePolicy {
    std::size_t batch_size{0};
    std::array<CnGeometry, 6> cn_geometries{};
    std::array<int, kCoreAlgorithmCount> core_threads{};
    double throughput_hps{0.0};
    bool from_cache{false};
};

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& geometry)
{
    launch_split_cryptonight_variant_production<VariantIndex>(stream, states, count,
                                                              scratchpads, contexts,
                                                              geometry);
}

void launch_split_cryptonight(cudaStream_t stream,
                              std::uint8_t* states,
                              std::size_t count,
                              std::uint8_t variant,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              const CnGeometry& geometry)
{
    switch (variant) {
    case 0: launch_split_cryptonight_variant<0>(stream, states, count, scratchpads, contexts, geometry); break;
    case 1: launch_split_cryptonight_variant<1>(stream, states, count, scratchpads, contexts, geometry); break;
    case 2: launch_split_cryptonight_variant<2>(stream, states, count, scratchpads, contexts, geometry); break;
    case 3: launch_split_cryptonight_variant<3>(stream, states, count, scratchpads, contexts, geometry); break;
    case 4: launch_split_cryptonight_variant<4>(stream, states, count, scratchpads, contexts, geometry); break;
    case 5: launch_split_cryptonight_variant<5>(stream, states, count, scratchpads, contexts, geometry); break;
    default: throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
    }
}

namespace {

int faststart_warp_aligned_down(int value, int warp)
{
    warp = std::max(1, warp);
    return (value / warp) * warp;
}

int faststart_core_threads(const cudaDeviceProp& props, int kernel_max)
{
    const int warp = std::max(1, props.warpSize);
    const int limit = std::max(warp, std::min(kernel_max, props.maxThreadsPerBlock));
    return std::max(warp, faststart_warp_aligned_down(std::min(kCoreDefaultThreads, limit), warp));
}

} // anonymous namespace

GpuTunePolicy build_gpu_tune_policy(int device_id,
                                    std::size_t requested_batch,
                                    std::size_t capacity,
                                    cudaStream_t,
                                    std::uint8_t*,
                                    std::uint8_t*,
                                    cryptonight::SplitContext*)
{
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties GPU fast-start failed");

    cudaFuncAttributes attrs{};
    check_cuda(cudaFuncGetAttributes(&attrs, core512_stage),
               "cudaFuncGetAttributes GPU fast-start failed");

    const int kernel_max = std::max(1, std::min(props.maxThreadsPerBlock, attrs.maxThreadsPerBlock));
    const int core_threads = faststart_core_threads(props, kernel_max);
    const int warp = std::max(1, props.warpSize);

    GpuTunePolicy policy{};
    policy.batch_size = std::max<std::size_t>(1U, std::min(requested_batch, capacity));
    policy.core_threads.fill(core_threads);
    for (auto& geometry : policy.cn_geometries) {
        geometry = CnGeometry{};
        geometry.setup_threads = warp;
        geometry.loop_threads = warp;
        geometry.final_threads = warp;
        geometry.aes_backend = 0;
    }

    std::cout << "[GPU " << device_id << "] production GPU fast-start"
              << " | batch=" << policy.batch_size
              << " | core=" << core_threads
              << " | CN=" << warp << '/' << warp << '/' << warp
              << " | source=runtime-safe"
              << " | startup-benchmark=off\n";
    return policy;
}
