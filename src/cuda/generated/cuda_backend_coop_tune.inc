// Per-device cooperative CryptoNight geometry sweep.
//
// This is still probe-only: it compares the parity-proven coop4 kernel against
// the actual tuned production single-hash loop, then sweeps cooperative block
// sizes so results are useful across NVIDIA architectures rather than being
// hard-coded for one Pascal card.

namespace {

template <std::uint8_t V>
void coop_tuned_benchmark_variant(int device_id,
                                  cudaStream_t stream,
                                  std::uint8_t* states,
                                  std::size_t n,
                                  std::uint8_t* scratch,
                                  cryptonight::SplitContext* ctx,
                                  const CnGeometry& g)
{
    auto parity_launch = [&](std::size_t c, std::uint8_t* s, cryptonight::SplitContext* x) {
        constexpr int threads = 32;
        const int blocks = static_cast<int>((c * 4U + threads - 1U) / threads);
        cryptonight_loop_stage_coop4<V><<<blocks, threads, 0, stream>>>(c, s, x);
    };
    auto v2_launch = [&](std::size_t c, std::uint8_t* s, cryptonight::SplitContext* x) {
        constexpr int threads = 32;
        const int blocks = static_cast<int>((c * 4U + threads - 1U) / threads);
        cryptonight_loop_stage_coop4_v2<V><<<blocks, threads, 0, stream>>>(c, s, x);
    };

    const bool coop_ok = coop_parity<V>(stream, g, states, parity_launch);
    const bool v2_ok = coop_parity<V>(stream, g, states, v2_launch);
    const int setup_threads = g.setup_threads;
    const int setup_blocks = static_cast<int>((n + static_cast<std::size_t>(setup_threads) - 1U) /
                                              static_cast<std::size_t>(setup_threads));

    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x63U + V * 13U),
                                   n * kStateBytes, stream),
                   "coop geometry prep states failed");
        if (g.aes_backend == 1)
            cryptonight_setup_stage<V, true><<<setup_blocks, setup_threads, 0, stream>>>(
                states, n, scratch, ctx);
        else
            cryptonight_setup_stage<V, false><<<setup_blocks, setup_threads, 0, stream>>>(
                states, n, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop geometry setup failed");
    };

    prepare();
    const float production_ms = time_cuda_phase(stream, [&]() {
        launch_cn_selected_single<V>(stream, device_id, g.loop_threads, n, scratch, ctx);
        check_cuda(cudaGetLastError(), "coop production baseline launch failed");
    });

    float best_ms = 1.0e30F;
    int best_threads = 0;
    if (coop_ok) {
        constexpr std::array<int, 4> candidates{{32, 64, 128, 256}};
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, device_id),
                   "cudaGetDeviceProperties cooperative geometry sweep failed");
        for (const int threads : candidates) {
            if (threads > props.maxThreadsPerBlock) continue;
            prepare();
            const int blocks = static_cast<int>((n * 4U + static_cast<std::size_t>(threads) - 1U) /
                                                static_cast<std::size_t>(threads));
            const float ms = time_cuda_phase(stream, [&]() {
                cryptonight_loop_stage_coop4<V><<<blocks, threads, 0, stream>>>(n, scratch, ctx);
                check_cuda(cudaGetLastError(), "coop geometry launch failed");
            });
            std::cout << std::fixed << std::setprecision(3)
                      << "[CUDA CN coop geometry] " << cryptonight::config_value(V).name
                      << " | hashes=" << n
                      << " | threads=" << threads
                      << " | " << ms << " ms"
                      << std::defaultfloat << '\n';
            if (ms < best_ms) {
                best_ms = ms;
                best_threads = threads;
            }
        }
    }

    const bool faster = coop_ok && best_threads != 0 && best_ms < production_ms;
    const float gain = faster ? (production_ms / best_ms - 1.0F) * 100.0F : 0.0F;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN coop result] " << cryptonight::config_value(V).name
              << " | production=" << production_ms << " ms"
              << " | coop4_parity=" << (coop_ok ? "PASS" : "FAIL")
              << " | coop4_best=" << (coop_ok ? best_ms : 0.0F) << " ms"
              << " | threads=" << best_threads
              << " | gain=" << gain << "%"
              << " | coop4-v2=" << (v2_ok ? "PASS" : "FAIL")
              << " | selected=production-probe-only"
              << std::defaultfloat << '\n';
}

} // anonymous namespace

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    auto result = autotune_cn_geometries_prod(device_id, stream, states, batch_size,
                                              scratchpads, contexts);
    const char* probe = std::getenv("YERBAS_CUDA_COOP_PROBE");
    if (!(probe && *probe && std::string(probe) != "0")) return result;

    std::cout << "[GPU " << device_id << "] cooperative CryptoNight geometry sweep starting"
              << " | hashes=" << batch_size << " | candidates=32,64,128,256\n";
    coop_tuned_benchmark_variant<0>(device_id, stream, states, batch_size, scratchpads, contexts, result[0]);
    coop_tuned_benchmark_variant<1>(device_id, stream, states, batch_size, scratchpads, contexts, result[1]);
    coop_tuned_benchmark_variant<2>(device_id, stream, states, batch_size, scratchpads, contexts, result[2]);
    coop_tuned_benchmark_variant<3>(device_id, stream, states, batch_size, scratchpads, contexts, result[3]);
    coop_tuned_benchmark_variant<4>(device_id, stream, states, batch_size, scratchpads, contexts, result[4]);
    coop_tuned_benchmark_variant<5>(device_id, stream, states, batch_size, scratchpads, contexts, result[5]);
    std::cout << "[GPU " << device_id << "] cooperative CryptoNight geometry sweep complete"
              << " | production unchanged\n";
    return result;
}
