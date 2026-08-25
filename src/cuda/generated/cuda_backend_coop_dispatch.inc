// Production dispatch wrapper for the parity-gated cooperative CryptoNight loop.
// The legacy split implementation remains the fallback for any variant/device
// where cooperative mode was not selected by the production autotuner.

namespace {

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& geometry)
{
    int device_id = -1;
    bool use_coop = false;
    int coop_threads = 32;
    int coop_impl = 1;
    if (cudaGetDevice(&device_id) == cudaSuccess &&
        device_id >= 0 && device_id < kMaxLoopTuneDevices &&
        g_cn_coop_ready[device_id] && !cn_coop_disabled_by_env()) {
        use_coop = g_cn_coop_enabled[device_id][VariantIndex];
        coop_threads = g_cn_coop_threads[device_id][VariantIndex];
        if (g_cn_coop_impl_ready[device_id])
            coop_impl = g_cn_coop_impl[device_id][VariantIndex];
    } else if (device_id < 0 || device_id >= kMaxLoopTuneDevices) {
        cudaGetLastError();
    }

    if (!use_coop) {
        launch_split_cryptonight_variant_legacy<VariantIndex>(stream, states, count,
                                                              scratchpads, contexts, geometry);
        return;
    }

    const int setup_threads = geometry.setup_threads;
    const int final_threads = geometry.final_threads;
    const int setup_blocks = static_cast<int>((count + static_cast<std::size_t>(setup_threads) - 1U) /
                                              static_cast<std::size_t>(setup_threads));
    const int final_blocks = static_cast<int>((count + static_cast<std::size_t>(final_threads) - 1U) /
                                              static_cast<std::size_t>(final_threads));

    if (geometry.aes_backend == 1)
        cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, setup_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA cooperative CryptoNight setup launch failed");

    launch_cn_coop_impl<VariantIndex>(stream, coop_impl, coop_threads, count,
                                      scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA cooperative CryptoNight loop launch failed");

    if (!g_cn_coop_runtime_reported[device_id][VariantIndex]) {
        g_cn_coop_runtime_reported[device_id][VariantIndex] = true;
        std::cout << "[CUDA CN runtime] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | backend=coop4-v" << coop_impl
                  << " | threads=" << coop_threads
                  << " | hashes=" << count << '\n';
    }

    if (geometry.aes_backend == 1)
        cryptonight_final_stage<VariantIndex, true><<<final_blocks, final_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    else
        cryptonight_final_stage<VariantIndex, false><<<final_blocks, final_threads, 0, stream>>>(
            states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA cooperative CryptoNight final launch failed");
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

} // anonymous namespace
