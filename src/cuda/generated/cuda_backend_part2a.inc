
__global__ void aes_ttable_validation_kernel(unsigned int* mismatches)
{
    __shared__ std::uint32_t aes_tables[1024];
    cryptonight::init_shared_aes_ttables(aes_tables);
    __syncthreads();

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 256U) return;

    alignas(16) std::uint8_t key[32];
    alignas(16) std::uint8_t expanded[240];
    alignas(16) std::uint8_t a[16];
    alignas(16) std::uint8_t b[16];
    #pragma unroll
    for (int i = 0; i < 32; ++i) key[i] = static_cast<std::uint8_t>((tid * 29U + i * 17U + 0x53U) & 0xffU);
    #pragma unroll
    for (int i = 0; i < 16; ++i) a[i] = b[i] = static_cast<std::uint8_t>((tid * 11U + i * 23U + 0xa7U) & 0xffU);
    cryptonight::aes256_expand_key(key, expanded);
    cryptonight::aes_pseudo_round(a, expanded);
    cryptonight::aes_pseudo_round_ttable(b, expanded, aes_tables);
    bool same = true;
    #pragma unroll
    for (int i = 0; i < 16; ++i) same = same && (a[i] == b[i]);
    if (!same) atomicAdd(mismatches, 1U);
}

bool validate_aes_ttable_backend(cudaStream_t stream)
{
    unsigned int* d_mismatches = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_mismatches), sizeof(unsigned int)),
               "cudaMalloc AES validation counter failed");
    try {
        check_cuda(cudaMemsetAsync(d_mismatches, 0, sizeof(unsigned int), stream),
                   "cudaMemsetAsync AES validation counter failed");
        aes_ttable_validation_kernel<<<1, 256, 0, stream>>>(d_mismatches);
        check_cuda(cudaGetLastError(), "AES T-table validation launch failed");
        unsigned int mismatches = 0;
        check_cuda(cudaMemcpyAsync(&mismatches, d_mismatches, sizeof(mismatches),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync AES validation result failed");
        check_cuda(cudaStreamSynchronize(stream), "AES T-table validation synchronize failed");
        cudaFree(d_mismatches);
        return mismatches == 0U;
    } catch (...) {
        cudaFree(d_mismatches);
        throw;
    }
}

template <std::uint8_t VariantIndex>
void benchmark_aes_backend(cudaStream_t stream,
                           std::uint8_t* states,
                           std::size_t sample_count,
                           std::uint8_t* scratchpads,
                           cryptonight::SplitContext* contexts,
                           CnGeometry& g,
                           bool ttable_valid)
{
    auto reset_states = [&]() {
        check_cuda(cudaMemsetAsync(states,
                                   static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes,
                                   stream),
                   "cudaMemsetAsync AES backend benchmark states failed");
    };
    const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                              static_cast<std::size_t>(g.setup_threads));
    const int loop_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.loop_threads) - 1U) /
                                             static_cast<std::size_t>(g.loop_threads));
    const int final_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(g.final_threads) - 1U) /
                                              static_cast<std::size_t>(g.final_threads));

    reset_states();
    const float portable_setup = time_cuda_phase(stream, [&]() {
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "portable AES setup benchmark launch failed");
    });
    reset_states();
    cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
    cryptonight_loop_stage<VariantIndex><<<loop_blocks, g.loop_threads, 0, stream>>>(sample_count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "portable AES final benchmark prep failed");
    const float portable_final = time_cuda_phase(stream, [&]() {
        cryptonight_final_stage<VariantIndex, false><<<final_blocks, g.final_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "portable AES final benchmark launch failed");
    });
    g.portable_aes_ms = portable_setup + portable_final;

    if (!ttable_valid) {
        g.ttable_aes_ms = g.portable_aes_ms;
        g.aes_backend = 0;
        return;
    }

    reset_states();
    const float ttable_setup = time_cuda_phase(stream, [&]() {
        cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "T-table AES setup benchmark launch failed");
    });
    reset_states();
    cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, g.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
    cryptonight_loop_stage<VariantIndex><<<loop_blocks, g.loop_threads, 0, stream>>>(sample_count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "T-table AES final benchmark prep failed");
    const float ttable_final = time_cuda_phase(stream, [&]() {
        cryptonight_final_stage<VariantIndex, true><<<final_blocks, g.final_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "T-table AES final benchmark launch failed");
    });
    g.ttable_aes_ms = ttable_setup + ttable_final;
    g.aes_backend = (g.ttable_aes_ms < g.portable_aes_ms) ? 1 : 0;

    // Keep the phase timings representative of the selected production backend.
    if (g.aes_backend == 1) {
        g.setup_ms = ttable_setup;
        g.final_ms = ttable_final;
    } else {
        g.setup_ms = portable_setup;
        g.final_ms = portable_final;
    }

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA AES autotune] " << cryptonight::config_value(VariantIndex).name
              << " | portable=" << g.portable_aes_ms << " ms"
              << " | ttable=" << g.ttable_aes_ms << " ms"
              << " | selected=" << (g.aes_backend == 1 ? "ttable" : "portable")
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex>
CnGeometry autotune_cn_variant(cudaStream_t stream,
                               std::uint8_t* states,
                               std::size_t sample_count,
                               std::uint8_t* scratchpads,
                               cryptonight::SplitContext* contexts,
                               int max_threads)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::array<int, 4> candidates{{32, 64, 128, 256}};
    CnGeometry best{};
    float best_setup_ms = 1.0e30F;
    float best_loop_ms = 1.0e30F;
    float best_final_ms = 1.0e30F;
    auto reset_states = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes, stream),
                   "cudaMemsetAsync CryptoNight autotune states failed");
    };
    for (const int threads : candidates) {
        if (threads > max_threads) continue;
        reset_states();
        const int blocks = static_cast<int>((sample_count + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        const float ms = time_cuda_phase(stream, [&]() {
            cryptonight_setup_stage<VariantIndex><<<blocks, threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
            check_cuda(cudaGetLastError(), "CryptoNight autotune setup launch failed");
        });
        if (ms < best_setup_ms) { best_setup_ms = ms; best.setup_threads = threads; }
    }
    for (const int threads : candidates) {
        if (threads > max_threads) continue;
        reset_states();
        const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(best.setup_threads) - 1U) /
                                                  static_cast<std::size_t>(best.setup_threads));
        cryptonight_setup_stage<VariantIndex><<<setup_blocks, best.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CryptoNight autotune loop prep failed");
        const int blocks = static_cast<int>((sample_count + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        const float ms = time_cuda_phase(stream, [&]() {
            cryptonight_loop_stage<VariantIndex><<<blocks, threads, 0, stream>>>(sample_count, scratchpads, contexts);
            check_cuda(cudaGetLastError(), "CryptoNight autotune loop launch failed");
        });
