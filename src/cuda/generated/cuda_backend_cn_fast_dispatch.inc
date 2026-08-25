// Public wrapper for the final CN-Fast family tuner.
// Preserve an existing production unroll-4 result because the focused challenger
// set intentionally compares only u1/u2, dual u1/u2, and coop4. If coop4 already
// won the broad production selector, continue into the final family comparison.

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    auto result = autotune_cn_geometries_coop(device_id, stream, states, batch_size,
                                              scratchpads, contexts);

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

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id),
               "cudaGetDeviceProperties CN-Fast family tuner failed");
    tune_cn_fast_family(device_id, props, stream, states, batch_size,
                        scratchpads, contexts, result[2]);

    const bool coop_selected = valid_device && g_cn_coop_ready[device_id] &&
                               g_cn_coop_enabled[device_id][2];
    if (!coop_selected) return result;

    // Normal startup first tries the per-device phase geometry learned by the
    // explicit benchmark. Forced benchmark/retune bypasses the cache, reruns the
    // A/B sweep, then stores whichever geometry remained selected after the 3%
    // whole-stage gate and parity check.
    if (load_cn_fast_micro_cache(device_id, props, batch_size, result[2]))
        return result;

    tune_cn_fast_microkernel(device_id, props, stream, states, batch_size,
                             scratchpads, contexts, result[2]);

    const char* benchmark = std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if (benchmark && *benchmark && std::string(benchmark) != "0") {
        int coop_threads = g_cn_coop_threads[device_id][2];
        if (coop_threads <= 0) coop_threads = 32;
        save_cn_fast_micro_cache(device_id, props, batch_size,
                                 result[2].setup_threads,
                                 coop_threads,
                                 result[2].final_threads);
    }
    return result;
}
