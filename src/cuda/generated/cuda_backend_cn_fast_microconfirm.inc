// Direct whole-stage confirmation for a CN-Fast microgeometry candidate.
// Phase-by-phase tuning can overstate a gain when measurements are noisy, so the
// selected geometry is re-run as setup -> coop4 loop -> final in one CUDA timing
// window. Three samples are taken and the median is used for the production gate.

namespace {

float benchmark_cn_fast_whole_once(cudaStream_t stream,
                                   std::uint8_t* states,
                                   std::size_t count,
                                   std::uint8_t* scratchpads,
                                   cryptonight::SplitContext* contexts,
                                   const CnGeometry& g,
                                   int setup_threads,
                                   int coop_threads,
                                   int final_threads)
{
    check_cuda(cudaMemsetAsync(states, 0x5b, count * kStateBytes, stream),
               "CN-Fast whole-stage state reset failed");
    return time_cuda_phase(stream, [&]() {
        const int setup_blocks = static_cast<int>((count + static_cast<std::size_t>(setup_threads) - 1U) /
                                                  static_cast<std::size_t>(setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<2, true><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_setup_stage<2, false><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast whole-stage setup failed");

        const int coop_blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(coop_threads) - 1U) /
                                                 static_cast<std::size_t>(coop_threads));
        cryptonight_loop_stage_coop4<2><<<coop_blocks, coop_threads, 0, stream>>>(count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast whole-stage coop4 failed");

        const int final_blocks = static_cast<int>((count + static_cast<std::size_t>(final_threads) - 1U) /
                                                  static_cast<std::size_t>(final_threads));
        if (g.aes_backend == 1)
            cryptonight_final_stage<2, true><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
        else
            cryptonight_final_stage<2, false><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CN-Fast whole-stage final failed");
    });
}

float benchmark_cn_fast_whole_median(cudaStream_t stream,
                                     std::uint8_t* states,
                                     std::size_t count,
                                     std::uint8_t* scratchpads,
                                     cryptonight::SplitContext* contexts,
                                     const CnGeometry& g,
                                     int setup_threads,
                                     int coop_threads,
                                     int final_threads)
{
    std::array<float, 3> samples{};
    for (float& sample : samples)
        sample = benchmark_cn_fast_whole_once(stream, states, count, scratchpads, contexts,
                                              g, setup_threads, coop_threads, final_threads);
    std::sort(samples.begin(), samples.end());
    return samples[1];
}

} // anonymous namespace
