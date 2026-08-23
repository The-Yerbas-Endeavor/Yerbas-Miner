// Experimental 4-lane cooperative CryptoNight loop probe.
//
// This does NOT alter the production backend. It parity-checks and benchmarks
// a 4-lane-per-hash AES-assisted loop against the known-good single-thread loop
// when YERBAS_CUDA_COOP_PROBE=1 is set. The goal is to measure whether subwarp
// cooperation can overcome the serial CryptoNight dependency chain on a given
// GPU before we wire it into production dispatch.

namespace {

__device__ __forceinline__ std::uint32_t coop_mix_column(std::uint8_t a0,
                                                         std::uint8_t a1,
                                                         std::uint8_t a2,
                                                         std::uint8_t a3)
{
    const std::uint8_t t = static_cast<std::uint8_t>(a0 ^ a1 ^ a2 ^ a3);
    const std::uint8_t o0 = static_cast<std::uint8_t>(a0 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a0 ^ a1)));
    const std::uint8_t o1 = static_cast<std::uint8_t>(a1 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a1 ^ a2)));
    const std::uint8_t o2 = static_cast<std::uint8_t>(a2 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a2 ^ a3)));
    const std::uint8_t o3 = static_cast<std::uint8_t>(a3 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a3 ^ a0)));
    return cryptonight::pack4(o0, o1, o2, o3);
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_coop4(std::size_t count,
                                             std::uint8_t* scratchpads,
                                             const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;

    const unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int group_base = warp_lane & ~3U;
    const unsigned int group = global_tid >> 2;
    if (group >= count) return;

    const unsigned int mask = 0xFU << group_base;
    std::uint8_t* scratchpad = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const cryptonight::SplitContext& ctx = contexts[group];

    alignas(16) std::uint8_t a[16]{};
    alignas(16) std::uint8_t b[16]{};
    alignas(16) std::uint8_t c[16]{};
    alignas(16) std::uint8_t tbuf[16]{};
    std::uint64_t tweak = 0;
    if (lane == 0U) {
        cryptonight::copy16(a, ctx.a);
        cryptonight::copy16(b, ctx.b);
        tweak = ctx.tweak;
    }

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        std::uint32_t aw0 = 0, aw1 = 0, aw2 = 0, aw3 = 0;
        std::uint32_t slot_index = 0;
        if (lane == 0U) {
            aw0 = cryptonight::load32le_aligned(a + 0);
            aw1 = cryptonight::load32le_aligned(a + 4);
            aw2 = cryptonight::load32le_aligned(a + 8);
            aw3 = cryptonight::load32le_aligned(a + 12);
            slot_index = static_cast<std::uint32_t>(cryptonight::cn_index_masked(a, address_mask));
        }
        aw0 = __shfl_sync(mask, aw0, group_base);
        aw1 = __shfl_sync(mask, aw1, group_base);
        aw2 = __shfl_sync(mask, aw2, group_base);
        aw3 = __shfl_sync(mask, aw3, group_base);
        slot_index = __shfl_sync(mask, slot_index, group_base);

        std::uint8_t* slot = scratchpad + static_cast<std::size_t>(slot_index) * 16U;
        std::uint8_t x0, x1, x2, x3;
        if (lane == 0U) {
            x0 = slot[0];  x1 = slot[5];  x2 = slot[10]; x3 = slot[15];
        } else if (lane == 1U) {
            x0 = slot[4];  x1 = slot[9];  x2 = slot[14]; x3 = slot[3];
        } else if (lane == 2U) {
            x0 = slot[8];  x1 = slot[13]; x2 = slot[2];  x3 = slot[7];
        } else {
            x0 = slot[12]; x1 = slot[1];  x2 = slot[6];  x3 = slot[11];
        }
        const std::uint32_t key_word = lane == 0U ? aw0 : lane == 1U ? aw1 : lane == 2U ? aw2 : aw3;
        const std::uint32_t cword = coop_mix_column(cryptonight::aes_sbox(x0),
                                                    cryptonight::aes_sbox(x1),
                                                    cryptonight::aes_sbox(x2),
                                                    cryptonight::aes_sbox(x3)) ^ key_word;
        // All four lanes must execute each _sync shuffle in the subgroup mask.
        const std::uint32_t c0 = __shfl_sync(mask, cword, group_base + 0U);
        const std::uint32_t c1 = __shfl_sync(mask, cword, group_base + 1U);
        const std::uint32_t c2 = __shfl_sync(mask, cword, group_base + 2U);
        const std::uint32_t c3 = __shfl_sync(mask, cword, group_base + 3U);

        if (lane == 0U) {
            cryptonight::store32le_aligned(c + 0, c0);
            cryptonight::store32le_aligned(c + 4, c1);
            cryptonight::store32le_aligned(c + 8, c2);
            cryptonight::store32le_aligned(c + 12, c3);

            reinterpret_cast<std::uint64_t*>(slot)[0] =
                reinterpret_cast<const std::uint64_t*>(c)[0] ^ reinterpret_cast<const std::uint64_t*>(b)[0];
            reinterpret_cast<std::uint64_t*>(slot)[1] =
                reinterpret_cast<const std::uint64_t*>(c)[1] ^ reinterpret_cast<const std::uint64_t*>(b)[1];
            cryptonight::variant1_mutate(slot);

            std::size_t j = cryptonight::cn_index_masked(c, address_mask);
            slot = scratchpad + j * 16U;
            cryptonight::copy16(tbuf, slot);
            const std::uint64_t c_lo = cryptonight::cn_load64_aligned(c);
            const std::uint64_t t0 = cryptonight::cn_load64_aligned(tbuf);
            const std::uint64_t lo = c_lo * t0;
            const std::uint64_t hi = __umul64hi(c_lo, t0);
            std::uint64_t a0 = cryptonight::cn_load64_aligned(a) + hi;
            std::uint64_t a1 = cryptonight::cn_load64_aligned(a + 8) + lo;
            cryptonight::cn_store64_aligned(slot, a0);
            cryptonight::cn_store64_aligned(slot + 8, a1 ^ tweak);
            a0 ^= t0;
            a1 ^= cryptonight::cn_load64_aligned(tbuf + 8);
            cryptonight::cn_store64_aligned(a, a0);
            cryptonight::cn_store64_aligned(a + 8, a1);
            cryptonight::copy16(b, c);
        }
        __syncwarp(mask);
    }
}

__global__ void compare_bytes_kernel(const std::uint8_t* a,
                                     const std::uint8_t* b,
                                     std::size_t bytes,
                                     unsigned int* mismatches)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes && a[i] != b[i]) atomicAdd(mismatches, 1U);
}

template <std::uint8_t VariantIndex>
bool coop4_parity_probe(cudaStream_t stream,
                        const CnGeometry& g,
                        std::uint8_t* states)
{
    constexpr std::size_t sample = 2;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    const std::size_t scratch_bytes = sample * static_cast<std::size_t>(cfg.page_size);
    std::uint8_t* ref_scratch = nullptr;
    std::uint8_t* coop_scratch = nullptr;
    cryptonight::SplitContext* ref_ctx = nullptr;
    cryptonight::SplitContext* coop_ctx = nullptr;
    unsigned int* mismatches = nullptr;
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ref_scratch), scratch_bytes), "coop parity ref scratch allocation failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&coop_scratch), scratch_bytes), "coop parity coop scratch allocation failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ref_ctx), sample * sizeof(*ref_ctx)), "coop parity ref context allocation failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&coop_ctx), sample * sizeof(*coop_ctx)), "coop parity coop context allocation failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&mismatches), sizeof(unsigned int)), "coop parity mismatch allocation failed");
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x47U + VariantIndex * 19U), sample * kStateBytes, stream),
                   "coop parity state prep failed");
        const int setup_threads = g.setup_threads;
        const int setup_blocks = static_cast<int>((sample + setup_threads - 1U) / setup_threads);
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(states, sample, ref_scratch, ref_ctx);
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(states, sample, coop_scratch, coop_ctx);
        check_cuda(cudaGetLastError(), "coop parity setup launch failed");
        launch_cn_loop_backend<VariantIndex>(stream, 1, false, 1, 32, sample, ref_scratch, ref_ctx);
        cryptonight_loop_stage_coop4<VariantIndex><<<1, 32, 0, stream>>>(sample, coop_scratch, coop_ctx);
        check_cuda(cudaGetLastError(), "coop parity loop launch failed");
        check_cuda(cudaMemsetAsync(mismatches, 0, sizeof(unsigned int), stream), "coop parity mismatch reset failed");
        const int cmp_threads = 256;
        const int cmp_blocks = static_cast<int>((scratch_bytes + cmp_threads - 1U) / cmp_threads);
        compare_bytes_kernel<<<cmp_blocks, cmp_threads, 0, stream>>>(ref_scratch, coop_scratch, scratch_bytes, mismatches);
        unsigned int host_mismatches = 0;
        check_cuda(cudaMemcpyAsync(&host_mismatches, mismatches, sizeof(host_mismatches), cudaMemcpyDeviceToHost, stream),
                   "coop parity result copy failed");
        check_cuda(cudaStreamSynchronize(stream), "coop parity synchronize failed");
        cudaFree(ref_scratch); cudaFree(coop_scratch); cudaFree(ref_ctx); cudaFree(coop_ctx); cudaFree(mismatches);
        return host_mismatches == 0U;
    } catch (...) {
        if (ref_scratch) cudaFree(ref_scratch);
        if (coop_scratch) cudaFree(coop_scratch);
        if (ref_ctx) cudaFree(ref_ctx);
        if (coop_ctx) cudaFree(coop_ctx);
        if (mismatches) cudaFree(mismatches);
        throw;
    }
}

template <std::uint8_t VariantIndex>
void coop4_benchmark_variant(int device_id,
                             cudaStream_t stream,
                             std::uint8_t* states,
                             std::size_t batch_size,
                             std::uint8_t* scratchpads,
                             cryptonight::SplitContext* contexts,
                             const CnGeometry& g)
{
    (void)device_id;
    const bool parity = coop4_parity_probe<VariantIndex>(stream, g, states);
    if (!parity) {
        std::cout << "[CUDA CN coop4 probe] " << cryptonight::config_value(VariantIndex).name
                  << " | parity=FAIL | selected=single\n";
        return;
    }
    const int setup_threads = g.setup_threads;
    const int setup_blocks = static_cast<int>((batch_size + static_cast<std::size_t>(setup_threads) - 1U) /
                                              static_cast<std::size_t>(setup_threads));
    auto prepare = [&]() {
        check_cuda(cudaMemsetAsync(states, static_cast<int>(0x63U + VariantIndex * 13U), batch_size * kStateBytes, stream),
                   "coop benchmark state prep failed");
        cryptonight_setup_stage<VariantIndex, false><<<setup_blocks, setup_threads, 0, stream>>>(
            states, batch_size, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "coop benchmark setup launch failed");
    };

    prepare();
    const int single_threads = g.loop_threads;
    const int single_blocks = static_cast<int>((batch_size + static_cast<std::size_t>(single_threads) - 1U) /
                                               static_cast<std::size_t>(single_threads));
    const float single_ms = time_cuda_phase(stream, [&]() {
        launch_cn_loop_backend<VariantIndex>(stream, 1, false, single_blocks, single_threads,
                                             batch_size, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "coop benchmark single launch failed");
    });

    prepare();
    constexpr int coop_threads = 128;
    const std::size_t coop_thread_count = batch_size * 4U;
    const int coop_blocks = static_cast<int>((coop_thread_count + coop_threads - 1U) / coop_threads);
    const float coop_ms = time_cuda_phase(stream, [&]() {
        cryptonight_loop_stage_coop4<VariantIndex><<<coop_blocks, coop_threads, 0, stream>>>(
            batch_size, scratchpads, contexts);
        check_cuda(cudaGetLastError(), "coop benchmark cooperative launch failed");
    });

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN coop4 probe] " << cryptonight::config_value(VariantIndex).name
              << " | hashes=" << batch_size
              << " | parity=PASS"
              << " | single=" << single_ms << " ms"
              << " | coop4=" << coop_ms << " ms"
              << " | ratio=" << (single_ms / coop_ms)
              << "x | selected=" << (coop_ms < single_ms * 0.98F ? "coop4-candidate" : "single")
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
    auto result = autotune_cn_geometries_prod(device_id, stream, states, batch_size, scratchpads, contexts);
    const char* probe = std::getenv("YERBAS_CUDA_COOP_PROBE");
    if (!(probe && *probe && std::string(probe) != "0")) return result;

    std::cout << "[GPU " << device_id << "] experimental CryptoNight coop4 probe starting | hashes="
              << batch_size << "\n";
    coop4_benchmark_variant<0>(device_id, stream, states, batch_size, scratchpads, contexts, result[0]);
    coop4_benchmark_variant<1>(device_id, stream, states, batch_size, scratchpads, contexts, result[1]);
    coop4_benchmark_variant<2>(device_id, stream, states, batch_size, scratchpads, contexts, result[2]);
    coop4_benchmark_variant<3>(device_id, stream, states, batch_size, scratchpads, contexts, result[3]);
    coop4_benchmark_variant<4>(device_id, stream, states, batch_size, scratchpads, contexts, result[4]);
    coop4_benchmark_variant<5>(device_id, stream, states, batch_size, scratchpads, contexts, result[5]);
    std::cout << "[GPU " << device_id << "] experimental CryptoNight coop4 probe complete | production unchanged\n";
    return result;
}
