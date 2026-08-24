// Public wrapper for the final CN-Fast family tuner.
// Preserve an existing production unroll-4 result because the focused challenger
// set intentionally compares only u1/u2, dual u1/u2, and coop4.

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    auto result = autotune_cn_geometries_coop(device_id, stream, states, batch_size,
                                              scratchpads, contexts);

    const bool preserve_u4 = device_id >= 0 && device_id < kMaxLoopTuneDevices &&
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
    return result;
}
