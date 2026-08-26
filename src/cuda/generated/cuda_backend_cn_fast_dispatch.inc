// Public wrapper for the final CN-Fast family tuner.
// Keep the broad production family selector and honor any previously qualified
// CN-Fast microgeometry cache. New microbenchmark generation has been retired.

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    auto result = autotune_cn_geometries_coop(device_id, stream, states, batch_size,
                                              scratchpads, contexts);

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties CN-Fast family tuner failed");

    ensure_cn_coop_impl_modes(device_id, props, stream, states, batch_size,
                              scratchpads, contexts, result);

    const bool valid_device = device_id >= 0 && device_id < kMaxLoopTuneDevices;
    const bool coop_active = valid_device && g_cn_coop_ready[device_id] &&
                             g_cn_coop_enabled[device_id][2];
    const bool preserve_u4 = valid_device && !coop_active &&
                             g_cn_loop_ready[device_id] &&
                             g_cn_loop_unroll[device_id][2] == 4;
    if (preserve_u4) {
        std::cout << "[CUDA CN-Fast autotune] GPU " << device_id
                  << " | existing production unroll4 preserved\n";
        return result;
    }

    tune_cn_fast_family(device_id, props, stream, states, batch_size,
                        scratchpads, contexts, result[2]);

    const bool coop_selected = valid_device && g_cn_coop_ready[device_id] &&
                               g_cn_coop_enabled[device_id][2];
    if (!coop_selected) return result;

    // Backward-compatible only: preserve a previously parity-qualified winner
    // from older builds, but do not generate new CN-Fast microbenchmark caches.
    (void)load_cn_fast_micro_cache(device_id, props, batch_size, result[2]);
    return result;
}
