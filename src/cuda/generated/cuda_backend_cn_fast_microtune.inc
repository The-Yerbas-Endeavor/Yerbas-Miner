// Focused CN-Fast coop4 microkernel + setup/final A/B tuner.
// Runs only when YERBAS_CUDA_CN_FAST_BENCHMARK=1 is set by --benchmark-cn-fast.
// It keeps the proven coop4 algorithm and sweeps launch geometry for the three
// CN-Fast phases independently.  A combined winner is parity-checked against
// the current production configuration and is adopted for the current run only
// when whole-stage time improves by at least 3%.

namespace {

constexpr float kCnFastMicroRequiredSpeedup = 0.97F;

bool cn_fast_micro_benchmark_enabled()
{
    const char* value = std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

float benchmark_cn_fast_setup_threads(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& g,
                                      int threads)
{
    check_cuda(cudaMemsetAsync(states, 0x6d, count * kStateBytes, stream),
               "CN-Fast micro setup state reset failed");
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    return time_cuda_phase(stream, [&]() {
        if (g.aes_backend == 1)
            cryptonight_setup_stage<2, true><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_setup_stage<2, false><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast micro setup benchmark failed");
    });
}

float benchmark_cn_fast_coop_threads(cudaStream_t stream,
                                     int threads,
                                     std::size_t count,
                                     std::uint8_t* scratchpads,
                                     cryptonight::SplitContext* contexts)
{
    const int blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    return time_cuda_phase(stream, [&]() {
        cryptonight_loop_stage_coop4<2><<<blocks, threads, 0, stream>>>(count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast micro coop4 benchmark failed");
    });
}

float benchmark_cn_fast_final_threads(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& g,
                                      int threads)
{
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    return time_cuda_phase(stream, [&]() {
        if (g.aes_backend == 1)
            cryptonight_final_stage<2, true><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_final_stage<2, false><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast micro final benchmark failed");
    });
}

void prepare_cn_fast_micro_for_loop(cudaStream_t stream,
                                    std::uint8_t* states,
                                    std::size_t count,
                                    std::uint8_t* scratchpads,
                                    cryptonight::SplitContext* contexts,
                                    const CnGeometry& g,
                                    int setup_threads)
{
    check_cuda(cudaMemsetAsync(states, 0x6d, count * kStateBytes, stream),
               "CN-Fast micro loop state reset failed");
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(setup_threads) - 1U) /
                                        static_cast<std::size_t>(setup_threads));
    if (g.aes_backend == 1)
        cryptonight_setup_stage<2, true><<<blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage<2, false><<<blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "CN-Fast micro loop preparation failed");
    check_cuda(cudaStreamSynchronize(stream), "CN-Fast micro loop preparation sync failed");
}

void prepare_cn_fast_micro_for_final(cudaStream_t stream,
                                     std::uint8_t* states,
                                     std::size_t count,
                                     std::uint8_t* scratchpads,
                                     cryptonight::SplitContext* contexts,
                                     const CnGeometry& g,
                                     int setup_threads,
                                     int coop_threads)
{
    prepare_cn_fast_micro_for_loop(stream, states, count, scratchpads, contexts, g, setup_threads);
    const int blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(coop_threads) - 1U) /
                                        static_cast<std::size_t>(coop_threads));
    cryptonight_loop_stage_coop4<2><<<blocks, coop_threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "CN-Fast micro final preparation loop failed");
    check_cuda(cudaStreamSynchronize(stream), "CN-Fast micro final preparation sync failed");
}

bool validate_cn_fast_micro_candidate(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& g,
                                      int baseline_setup,
                                      int baseline_coop,
                                      int baseline_final,
                                      int candidate_setup,
                                      int candidate_coop,
                                      int candidate_final)
{
    constexpr std::size_t count = 2U;
    std::array<std::uint8_t, count * kStateBytes> baseline{};
    std::array<std::uint8_t, count * kStateBytes> candidate{};

    auto run = [&](int setup_threads, int coop_threads, int final_threads,
                   std::array<std::uint8_t, count * kStateBytes>& output) {
        check_cuda(cudaMemsetAsync(states, 0x51, count * kStateBytes, stream),
                   "CN-Fast micro parity state reset failed");
        const int setup_blocks = static_cast<int>((count + static_cast<std::size_t>(setup_threads) - 1U) /
                                                  static_cast<std::size_t>(setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<2, true><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_setup_stage<2, false><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast micro parity setup failed");

        const int coop_blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(coop_threads) - 1U) /
                                                 static_cast<std::size_t>(coop_threads));
        cryptonight_loop_stage_coop4<2><<<coop_blocks, coop_threads, 0, stream>>>(count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast micro parity loop failed");

        const int final_blocks = static_cast<int>((count + static_cast<std::size_t>(final_threads) - 1U) /
                                                  static_cast<std::size_t>(final_threads));
        if (g.aes_backend == 1)
            cryptonight_final_stage<2, true><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_final_stage<2, false><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast micro parity final failed");
        check_cuda(cudaMemcpyAsync(output.data(), states, output.size(), cudaMemcpyDeviceToHost, stream),
                   "CN-Fast micro parity copy failed");
        check_cuda(cudaStreamSynchronize(stream), "CN-Fast micro parity sync failed");
    };

    run(baseline_setup, baseline_coop, baseline_final, baseline);
    run(candidate_setup, candidate_coop, candidate_final, candidate);
    return baseline == candidate;
}

void tune_cn_fast_microkernel(int device_id,
                              const cudaDeviceProp& props,
                              cudaStream_t stream,
                              std::uint8_t* states,
                              std::size_t batch_size,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts,
                              CnGeometry& g)
{
    if (!cn_fast_micro_benchmark_enabled() || batch_size == 0 ||
        device_id < 0 || device_id >= kMaxLoopTuneDevices)
        return;

    constexpr std::array<int, 4> candidates{{32, 64, 128, 256}};
    const int baseline_setup = g.setup_threads;
    const int baseline_final = g.final_threads;
    int baseline_coop = g_cn_coop_threads[device_id][2];
    if (baseline_coop <= 0) baseline_coop = 32;

    // Baseline phase timings using the exact current production geometry.
    const float baseline_setup_ms = benchmark_cn_fast_setup_threads(
        stream, states, batch_size, scratchpads, contexts, g, baseline_setup);
    prepare_cn_fast_micro_for_loop(stream, states, batch_size, scratchpads, contexts, g, baseline_setup);
    const float baseline_loop_ms = benchmark_cn_fast_coop_threads(
        stream, baseline_coop, batch_size, scratchpads, contexts);
    prepare_cn_fast_micro_for_final(stream, states, batch_size, scratchpads, contexts,
                                    g, baseline_setup, baseline_coop);
    const float baseline_final_ms = benchmark_cn_fast_final_threads(
        stream, states, batch_size, scratchpads, contexts, g, baseline_final);
    const float baseline_total_ms = baseline_setup_ms + baseline_loop_ms + baseline_final_ms;

    int best_setup = baseline_setup;
    float best_setup_ms = baseline_setup_ms;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN-Fast phase A/B] GPU " << device_id
              << " | baseline setup=" << baseline_setup << ':' << baseline_setup_ms << " ms"
              << " | coop=" << baseline_coop << ':' << baseline_loop_ms << " ms"
              << " | final=" << baseline_final << ':' << baseline_final_ms << " ms"
              << " | total=" << baseline_total_ms << " ms\n";

    for (const int threads : candidates) {
        if (threads > props.maxThreadsPerBlock) continue;
        const float ms = benchmark_cn_fast_setup_threads(
            stream, states, batch_size, scratchpads, contexts, g, threads);
        std::cout << "[CUDA CN-Fast setup A/B] GPU " << device_id
                  << " | threads=" << threads << " | ms=" << ms
                  << " | gain=" << (baseline_setup_ms / ms - 1.0F) * 100.0F << "%\n";
        if (ms < best_setup_ms) { best_setup_ms = ms; best_setup = threads; }
    }

    int best_coop = baseline_coop;
    float best_loop_ms = baseline_loop_ms;
    for (const int threads : candidates) {
        if (threads > props.maxThreadsPerBlock || (threads & 31) != 0) continue;
        prepare_cn_fast_micro_for_loop(stream, states, batch_size, scratchpads, contexts, g, best_setup);
        const float ms = benchmark_cn_fast_coop_threads(
            stream, threads, batch_size, scratchpads, contexts);
        std::cout << "[CUDA CN-Fast coop4 A/B] GPU " << device_id
                  << " | threads=" << threads << " | loop=" << ms << " ms"
                  << " | loop_hps=" << (static_cast<double>(batch_size) * 1000.0 / ms)
                  << " | gain=" << (baseline_loop_ms / ms - 1.0F) * 100.0F << "%\n";
        if (ms < best_loop_ms) { best_loop_ms = ms; best_coop = threads; }
    }

    int best_final = baseline_final;
    float best_final_ms = baseline_final_ms;
    for (const int threads : candidates) {
        if (threads > props.maxThreadsPerBlock) continue;
        prepare_cn_fast_micro_for_final(stream, states, batch_size, scratchpads, contexts,
                                        g, best_setup, best_coop);
        const float ms = benchmark_cn_fast_final_threads(
            stream, states, batch_size, scratchpads, contexts, g, threads);
        std::cout << "[CUDA CN-Fast final A/B] GPU " << device_id
                  << " | threads=" << threads << " | ms=" << ms
                  << " | gain=" << (baseline_final_ms / ms - 1.0F) * 100.0F << "%\n";
        if (ms < best_final_ms) { best_final_ms = ms; best_final = threads; }
    }

    const float candidate_total_ms = best_setup_ms + best_loop_ms + best_final_ms;
    const float whole_gain = (baseline_total_ms / candidate_total_ms - 1.0F) * 100.0F;
    const bool parity = validate_cn_fast_micro_candidate(
        stream, states, scratchpads, contexts, g,
        baseline_setup, baseline_coop, baseline_final,
        best_setup, best_coop, best_final);
    const bool qualified = parity && candidate_total_ms < baseline_total_ms * kCnFastMicroRequiredSpeedup;

    std::cout << "[CUDA CN-Fast micro winner] GPU " << device_id
              << " | setup=" << best_setup << ':' << best_setup_ms << " ms"
              << " | coop=" << best_coop << ':' << best_loop_ms << " ms"
              << " | final=" << best_final << ':' << best_final_ms << " ms"
              << " | total=" << candidate_total_ms << " ms"
              << " | gain=" << whole_gain << "%"
              << " | parity=" << (parity ? "PASS" : "FAIL")
              << " | selected=" << (qualified ? "candidate" : "baseline")
              << std::defaultfloat << '\n';

    if (qualified) {
        g.setup_threads = best_setup;
        g.final_threads = best_final;
        g_cn_coop_threads[device_id][2] = best_coop;
    }
}

} // anonymous namespace
