from pathlib import Path

p = Path('src/cuda/cuda_backend.cu')
s = p.read_text()

def rep(old, new, label):
    global s
    if old not in s:
        raise SystemExit(f'patch failed: {label}')
    s = s.replace(old, new, 1)

rep('constexpr int kCnGeometryCacheRevision = 3;',
    'constexpr int kCnGeometryCacheRevision = 4;', 'cache revision')

rep(r'''struct CnGeometry {
    int setup_threads{64};
    int loop_threads{64};
    int final_threads{64};
    float setup_ms{0.0F};
    float loop_ms{0.0F};
    float final_ms{0.0F};
};''',
r'''struct CnGeometry {
    int setup_threads{64};
    int loop_threads{64};
    int final_threads{64};
    int aes_backend{0}; // 0=portable, 1=shared T-table
    float setup_ms{0.0F};
    float loop_ms{0.0F};
    float final_ms{0.0F};
    float portable_aes_ms{0.0F};
    float ttable_aes_ms{0.0F};
};''', 'geometry fields')

rep(r'''        std::cout << "[CUDA CN profile] " << cryptonight::kVariantConfigs[i].name
                  << " | geometry=" << g.setup_threads << '/' << g.loop_threads << '/' << g.final_threads
                  << " | setup=" << g.setup_ms << " ms"
                  << " | loop=" << g.loop_ms << " ms"
                  << " | final=" << g.final_ms << " ms"
                  << " | total=" << total << " ms"
                  << " | loop=" << std::fixed << std::setprecision(1) << loop_pct << "%"
                  << std::defaultfloat << '\n';''',
r'''        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA CN profile] " << cryptonight::kVariantConfigs[i].name
                  << " | AES=" << (g.aes_backend == 1 ? "ttable" : "portable")
                  << " | geometry=" << g.setup_threads << '/' << g.loop_threads << '/' << g.final_threads
                  << " | setup=" << g.setup_ms << " ms"
                  << " | loop=" << g.loop_ms << " ms"
                  << " | final=" << g.final_ms << " ms"
                  << " | total=" << total << " ms"
                  << " | loop=" << std::setprecision(1) << loop_pct << "%"
                  << std::defaultfloat << '\n';''', 'fixed profile output')

rep(r'''        if (!(in >> variant >> g.setup_threads >> g.loop_threads >> g.final_threads
                 >> g.setup_ms >> g.loop_ms >> g.final_ms) ||''',
r'''        if (!(in >> variant >> g.setup_threads >> g.loop_threads >> g.final_threads >> g.aes_backend
                 >> g.setup_ms >> g.loop_ms >> g.final_ms >> g.portable_aes_ms >> g.ttable_aes_ms) ||''', 'cache read fields')

rep(r'''            !valid_cn_threads(g.final_threads, props.maxThreadsPerBlock) ||
            g.setup_ms <= 0.0F || g.loop_ms <= 0.0F || g.final_ms <= 0.0F) {''',
r'''            !valid_cn_threads(g.final_threads, props.maxThreadsPerBlock) ||
            (g.aes_backend != 0 && g.aes_backend != 1) ||
            g.setup_ms <= 0.0F || g.loop_ms <= 0.0F || g.final_ms <= 0.0F ||
            g.portable_aes_ms <= 0.0F || g.ttable_aes_ms <= 0.0F) {''', 'cache validation')

rep(r"""            out << i << ' ' << g.setup_threads << ' ' << g.loop_threads << ' ' << g.final_threads
                << ' ' << g.setup_ms << ' ' << g.loop_ms << ' ' << g.final_ms << '\n';""",
r"""            out << i << ' ' << g.setup_threads << ' ' << g.loop_threads << ' ' << g.final_threads
                << ' ' << g.aes_backend
                << ' ' << g.setup_ms << ' ' << g.loop_ms << ' ' << g.final_ms
                << ' ' << g.portable_aes_ms << ' ' << g.ttable_aes_ms << '\n';""", 'cache write fields')

rep(r'''template <std::uint8_t VariantIndex>
__global__ void cryptonight_setup_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    cryptonight::split_setup<VariantIndex>(state, kStateBytes, scratchpad, contexts[index]);
}
''',
r'''template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_setup_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    cryptonight::split_setup<VariantIndex, UseTTable>(state, kStateBytes, scratchpad, contexts[index],
                                                      UseTTable ? aes_tables : nullptr);
}
''', 'setup kernel')

rep(r'''template <std::uint8_t VariantIndex>
__global__ void cryptonight_final_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    std::uint8_t digest[32];
    cryptonight::split_finalize<VariantIndex>(scratchpad, contexts[index], digest);
    #pragma unroll
    for (int i = 0; i < 32; ++i) state[i] = digest[i];
    #pragma unroll
    for (int i = 32; i < 64; ++i) state[i] = 0;
}
''',
r'''template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_final_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    std::uint8_t digest[32];
    cryptonight::split_finalize<VariantIndex, UseTTable>(scratchpad, contexts[index], digest,
                                                         UseTTable ? aes_tables : nullptr);
    #pragma unroll
    for (int i = 0; i < 32; ++i) state[i] = digest[i];
    #pragma unroll
    for (int i = 32; i < 64; ++i) state[i] = 0;
}
''', 'final kernel')

insert = r'''__global__ void aes_ttable_validation_kernel(unsigned int* mismatches)
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
        check_cuda(cudaMemcpyAsync(&mismatches, d_mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost, stream),
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
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x31U + VariantIndex * 17U),
                                   sample_count * kStateBytes, stream),
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

'''
needle = 'template <std::uint8_t VariantIndex>\nCnGeometry autotune_cn_variant'
if needle not in s:
    raise SystemExit('patch failed: autotune insertion point')
s = s.replace(needle, insert + needle, 1)

rep(r'''    std::cout << "[CUDA autotune] " << cfg.name
              << " | sample=" << sample_count
              << " | setup=" << best.setup_threads << " (" << best.setup_ms << " ms)"
              << " | loop=" << best.loop_threads << " (" << best.loop_ms << " ms)"
              << " | final=" << best.final_threads << " (" << best.final_ms << " ms)\n";''',
r'''    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA autotune] " << cfg.name
              << " | sample=" << sample_count
              << " | setup=" << best.setup_threads << " (" << best.setup_ms << " ms)"
              << " | loop=" << best.loop_threads << " (" << best.loop_ms << " ms)"
              << " | final=" << best.final_threads << " (" << best.final_ms << " ms)"
              << std::defaultfloat << '\n';''', 'autotune decimals')

rep(r'''    std::cout << "[GPU " << device_id << "] empirical CryptoNight geometry autotune complete\n";

    print_cn_phase_measurements(device_id, result, sample_count, "fresh-autotune");''',
r'''    std::cout << "[GPU " << device_id << "] empirical CryptoNight geometry autotune complete\n";
    const bool ttable_valid = validate_aes_ttable_backend(stream);
    std::cout << "[GPU " << device_id << "] shared T-table AES parity validation: "
              << (ttable_valid ? "PASS" : "FAIL - portable AES forced") << '\n';
    benchmark_aes_backend<0>(stream, states, sample_count, scratchpads, contexts, result[0], ttable_valid);
    benchmark_aes_backend<1>(stream, states, sample_count, scratchpads, contexts, result[1], ttable_valid);
    benchmark_aes_backend<2>(stream, states, sample_count, scratchpads, contexts, result[2], ttable_valid);
    benchmark_aes_backend<3>(stream, states, sample_count, scratchpads, contexts, result[3], ttable_valid);
    benchmark_aes_backend<4>(stream, states, sample_count, scratchpads, contexts, result[4], ttable_valid);
    benchmark_aes_backend<5>(stream, states, sample_count, scratchpads, contexts, result[5], ttable_valid);
    print_cn_phase_measurements(device_id, result, sample_count, "fresh-autotune");''', 'AES benchmark selection')

rep(r'''    cryptonight_setup_stage<VariantIndex><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight setup launch failed");
    cryptonight_loop_stage<VariantIndex><<<loop_blocks, loop_threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight loop launch failed");
    cryptonight_final_stage<VariantIndex><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight final launch failed");''',
r'''    if (geometry.aes_backend == 1)
        cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight setup launch failed");
    cryptonight_loop_stage<VariantIndex><<<loop_blocks, loop_threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight loop launch failed");
    if (geometry.aes_backend == 1)
        cryptonight_final_stage<VariantIndex, true><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_final_stage<VariantIndex, false><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight final launch failed");''', 'production AES selection')

p.write_text(s)
print('cuda_backend.cu AES backend patch applied')
