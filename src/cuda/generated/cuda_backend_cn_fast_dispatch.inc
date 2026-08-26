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

    // Narrow CN-Fast experiments remain generic and cumulative: each new
    // candidate is compared against the best already-qualified path.
    ensure_cn_fast_v4_mode(device_id, props, stream, states, batch_size,
                           scratchpads, contexts, result[2]);
    ensure_cn_fast_v5_mode(device_id, props, stream, states, batch_size,
                           scratchpads, contexts, result[2]);
    ensure_cn_fast_v6_mode(device_id, props, stream, states, batch_size,
                           scratchpads, contexts, result[2]);

    if (load_cn_fast_micro_cache(device_id, props, batch_size, result[2]))
        return result;

    const int baseline_setup = result[2].setup_threads;
    const int baseline_final = result[2].final_threads;
    int baseline_coop = g_cn_coop_threads[device_id][2];
    if (baseline_coop <= 0) baseline_coop = 32;

    tune_cn_fast_microkernel(device_id, props, stream, states, batch_size,
                             scratchpads, contexts, result[2]);

    const char* benchmark = std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if (benchmark && *benchmark && std::string(benchmark) != "0") {
        int candidate_coop = g_cn_coop_threads[device_id][2];
        if (candidate_coop <= 0) candidate_coop = 32;
        const int candidate_setup = result[2].setup_threads;
        const int candidate_final = result[2].final_threads;
        const bool changed = candidate_setup != baseline_setup ||
                             candidate_coop != baseline_coop ||
                             candidate_final != baseline_final;

        bool direct_qualified = false;
        float baseline_ms = 0.0F;
        float candidate_ms = 0.0F;
        float gain = 0.0F;
        if (changed) {
            baseline_ms = benchmark_cn_fast_whole_median(
                stream, states, batch_size, scratchpads, contexts, result[2],
                baseline_setup, baseline_coop, baseline_final);
            candidate_ms = benchmark_cn_fast_whole_median(
                stream, states, batch_size, scratchpads, contexts, result[2],
                candidate_setup, candidate_coop, candidate_final);
            gain = candidate_ms > 0.0F ? (baseline_ms / candidate_ms - 1.0F) * 100.0F : 0.0F;
            direct_qualified = candidate_ms > 0.0F && candidate_ms < baseline_ms * 0.97F;

            std::cout << std::fixed << std::setprecision(3)
                      << "[CUDA CN-Fast whole A/B] GPU " << device_id
                      << " | baseline=" << baseline_setup << '/' << baseline_coop << '/' << baseline_final
                      << ':' << baseline_ms << " ms"
                      << " | candidate=" << candidate_setup << '/' << candidate_coop << '/' << candidate_final
                      << ':' << candidate_ms << " ms"
                      << " | gain=" << gain << "%"
                      << " | selected=" << (direct_qualified ? "candidate" : "baseline")
                      << std::defaultfloat << '\n';
        }

        if (changed && !direct_qualified) {
            result[2].setup_threads = baseline_setup;
            result[2].final_threads = baseline_final;
            g_cn_coop_threads[device_id][2] = baseline_coop;
        }

        int selected_coop = g_cn_coop_threads[device_id][2];
        if (selected_coop <= 0) selected_coop = 32;
        save_cn_fast_micro_cache(device_id, props, batch_size,
                                 result[2].setup_threads,
                                 selected_coop,
                                 result[2].final_threads);

        profile_cn_fast_coop4_inner_v2(device_id, stream, states, batch_size,
                                       scratchpads, contexts, result[2]);
    }
    return result;
}
