        if (ms < best_loop_ms) { best_loop_ms = ms; best.loop_threads = threads; }
    }
    for (const int threads : candidates) {
        if (threads > max_threads) continue;
        reset_states();
        const int setup_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(best.setup_threads) - 1U) /
                                                  static_cast<std::size_t>(best.setup_threads));
        const int loop_blocks = static_cast<int>((sample_count + static_cast<std::size_t>(best.loop_threads) - 1U) /
                                                 static_cast<std::size_t>(best.loop_threads));
        cryptonight_setup_stage<VariantIndex><<<setup_blocks, best.setup_threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CryptoNight autotune final setup prep failed");
        cryptonight_loop_stage<VariantIndex><<<loop_blocks, best.loop_threads, 0, stream>>>(sample_count, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "CryptoNight autotune final loop prep failed");
        const int blocks = static_cast<int>((sample_count + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        const float ms = time_cuda_phase(stream, [&]() {
            cryptonight_final_stage<VariantIndex><<<blocks, threads, 0, stream>>>(states, sample_count, scratchpads, contexts);
            check_cuda(cudaGetLastError(), "CryptoNight autotune final launch failed");
        });
        if (ms < best_final_ms) { best_final_ms = ms; best.final_threads = threads; }
    }
    best.setup_ms = best_setup_ms;
    best.loop_ms = best_loop_ms;
    best.final_ms = best_final_ms;
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA autotune] " << cfg.name
              << " | sample=" << sample_count
              << " | setup=" << best.setup_threads << " (" << best.setup_ms << " ms)"
              << " | loop=" << best.loop_threads << " (" << best.loop_ms << " ms)"
              << " | final=" << best.final_threads << " (" << best.final_ms << " ms)"
              << std::defaultfloat << '\n';
    return best;
}

std::array<CnGeometry, 6> autotune_cn_geometries(int device_id,
                                                 cudaStream_t stream,
                                                 std::uint8_t* states,
                                                 std::size_t batch_size,
                                                 std::uint8_t* scratchpads,
                                                 cryptonight::SplitContext* contexts)
{
    std::array<CnGeometry, 6> result{};
    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CryptoNight autotune failed");
    std::size_t sample_count = std::max<std::size_t>(32U,
        static_cast<std::size_t>(std::max(1, props.multiProcessorCount)) * 2U);
    sample_count = std::min(sample_count, batch_size);
    sample_count = std::max<std::size_t>(1U, sample_count);

    if (load_cn_geometry_cache(device_id, props, batch_size, sample_count, result)) {
        ensure_cn_loop_modes(device_id, props, stream, states, sample_count, batch_size,
                             scratchpads, contexts, result);
        return result;
    }

    std::cout << "[GPU " << device_id << "] empirical CryptoNight geometry autotune starting"
              << " | sample=" << sample_count << " | candidates=32,64,128,256\n";
    result[0] = autotune_cn_variant<0>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[1] = autotune_cn_variant<1>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[2] = autotune_cn_variant<2>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[3] = autotune_cn_variant<3>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[4] = autotune_cn_variant<4>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    result[5] = autotune_cn_variant<5>(stream, states, sample_count, scratchpads, contexts, props.maxThreadsPerBlock);
    std::cout << "[GPU " << device_id << "] empirical CryptoNight geometry autotune complete\n";

    const bool ttable_valid = validate_aes_ttable_backend(stream);
    std::cout << "[GPU " << device_id << "] shared T-table AES parity validation: "
              << (ttable_valid ? "PASS" : "FAIL - portable AES forced") << '\n';
    benchmark_aes_backend<0>(stream, states, sample_count, scratchpads, contexts, result[0], ttable_valid);
    benchmark_aes_backend<1>(stream, states, sample_count, scratchpads, contexts, result[1], ttable_valid);
    benchmark_aes_backend<2>(stream, states, sample_count, scratchpads, contexts, result[2], ttable_valid);
    benchmark_aes_backend<3>(stream, states, sample_count, scratchpads, contexts, result[3], ttable_valid);
    benchmark_aes_backend<4>(stream, states, sample_count, scratchpads, contexts, result[4], ttable_valid);
    benchmark_aes_backend<5>(stream, states, sample_count, scratchpads, contexts, result[5], ttable_valid);

    ensure_cn_loop_modes(device_id, props, stream, states, sample_count, batch_size,
                         scratchpads, contexts, result);
    print_cn_phase_measurements(device_id, result, sample_count, "fresh-autotune");
    save_cn_geometry_cache(device_id, props, batch_size, sample_count, result);
    return result;
}

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts,
                                      const CnGeometry& geometry)
{
    const int setup_threads = geometry.setup_threads;
    const int loop_threads = geometry.loop_threads;
    const int final_threads = geometry.final_threads;
    const int setup_blocks = static_cast<int>((count + setup_threads - 1) / setup_threads);
    const int loop_blocks = static_cast<int>((count + loop_threads - 1) / loop_threads);
    const int final_blocks = static_cast<int>((count + final_threads - 1) / final_threads);
    if (geometry.aes_backend == 1)
        cryptonight_setup_stage<VariantIndex, true><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight setup launch failed");

    int device_id = -1;
    int loop_mode = 1;
    if (cudaGetDevice(&device_id) == cudaSuccess && device_id >= 0 && device_id < kMaxLoopTuneDevices &&
        g_cn_loop_ready[device_id]) {
        loop_mode = g_cn_loop_unroll[device_id][VariantIndex];
    } else {
        cudaGetLastError();
    }
    launch_cn_loop_mode<VariantIndex>(stream, loop_mode, loop_blocks, loop_threads,
                                      count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight loop launch failed");

    if (geometry.aes_backend == 1)
        cryptonight_final_stage<VariantIndex, true><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_final_stage<VariantIndex, false><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight final launch failed");
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

__global__ void keccak512_validation_kernel(const std::uint8_t* input, std::size_t length, std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::keccak512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void cubehash512_validation_kernel(const std::uint8_t* input, std::size_t length, std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::cubehash512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void collect_candidates(const std::uint32_t* nonces,
                                   const std::uint8_t* states,
                                   std::size_t count,
                                   const DeviceJob* job,
                                   DeviceCandidate* candidates,
                                   unsigned int* candidate_count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint8_t* hash = states + index * kStateBytes;
    if (!hash_meets_target(hash, job->target_le)) return;
    const unsigned int slot = atomicAdd(candidate_count, 1U);
    if (slot >= kMaxCandidates) return;
    candidates[slot].nonce = nonces[index];
    for (int i = 0; i < 32; ++i) candidates[slot].hash[i] = hash[i];
}

Hash512 run_validation_kernel(int device_id,
                              const std::uint8_t* input,
                              std::size_t length,
                              bool use_keccak)
{
    if (input == nullptr || length == 0)
        throw std::invalid_argument("CUDA core validation input must not be empty");
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), length), "cudaMalloc validation input failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), 64), "cudaMalloc validation output failed");
        check_cuda(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice), "cudaMemcpy validation input failed");
        if (use_keccak) keccak512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        else cubehash512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        check_cuda(cudaGetLastError(), "CUDA core validation launch failed");
        check_cuda(cudaDeviceSynchronize(), "CUDA core validation synchronize failed");
        Hash512 out{};
        check_cuda(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost), "cudaMemcpy validation output failed");
        cudaFree(d_output);
        cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_output) cudaFree(d_output);
        if (d_input) cudaFree(d_input);
        throw;
    }
}

} // namespace

