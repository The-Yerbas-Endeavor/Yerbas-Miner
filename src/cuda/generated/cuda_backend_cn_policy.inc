// Single generic CryptoNight production policy.
// CN-Fast-specific family/microgeometry tuning has been retired; all six
// variants use the same cooperative qualification path and parity-gated
// coop4 implementation selector.

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
               "cudaGetDeviceProperties CN policy failed");
    ensure_cn_coop_impl_modes(device_id, props, stream, states, batch_size,
                              scratchpads, contexts, result);

    std::cout << "[GPU " << device_id << "] CN policy"
              << " | backend=generic-coop4"
              << " | variants=6"
              << " | cn-fast-special=off\n";
    return result;
}
