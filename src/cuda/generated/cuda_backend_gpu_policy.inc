// Single production-facing GPU tuning policy.
// BatchEngine consumes only this object. Individual benchmark helpers remain
// implementation details while their old cache layers are retired/consolidated.

struct GpuTunePolicy {
    std::size_t batch_size{0};
    std::array<CnGeometry, 6> cn_geometries{};
    std::array<int, kCoreAlgorithmCount> core_threads{};
};

GpuTunePolicy build_gpu_tune_policy(int device_id,
                                    std::size_t requested_batch,
                                    std::size_t capacity,
                                    cudaStream_t stream,
                                    std::uint8_t* states,
                                    std::uint8_t* scratchpads,
                                    cryptonight::SplitContext* contexts)
{
    GpuTunePolicy policy{};
    policy.core_threads.fill(kCoreDefaultThreads);
    policy.batch_size = autotune_cuda_active_batch(device_id, requested_batch, capacity,
                                                   stream, states, scratchpads, contexts);

    try {
        policy.cn_geometries = autotune_cn_geometries(device_id, stream, states,
                                                      policy.batch_size, scratchpads, contexts);
    } catch (const std::exception& e) {
        policy.cn_geometries.fill(CnGeometry{});
        cudaGetLastError();
        std::cerr << "[GPU " << device_id << "] CN policy failed; using safe 64/64/64: "
                  << e.what() << '\n';
    }

    try {
        policy.core_threads = autotune_core_threads(device_id, stream, states,
                                                    policy.batch_size);
    } catch (const std::exception& e) {
        policy.core_threads.fill(kCoreDefaultThreads);
        cudaGetLastError();
        std::cerr << "[GPU " << device_id << "] core policy failed; using 256 threads: "
                  << e.what() << '\n';
    }

    std::cout << "[GPU " << device_id << "] production tune policy"
              << " | batch=" << policy.batch_size
              << " | CN=generic"
              << " | cores=per-algorithm"
              << " | policy=gpu-v1\n";
    return policy;
}
