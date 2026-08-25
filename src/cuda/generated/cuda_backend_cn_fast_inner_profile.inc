// Sampled inner-loop profiler for the established coop4-v1 CN-Fast kernel.
//
// Enabled only by the explicit CN-Fast benchmark path. One cooperative hash
// group samples clock64() every 1024 iterations while the full production batch
// runs normally, keeping measurement overhead small and preserving realistic
// occupancy/memory pressure. Reported cycle shares are diagnostic, not used for
// production dispatch.

namespace {

__global__ void cryptonight_cn_fast_coop4_v1_profile(std::size_t count,
                                                     std::uint8_t* scratchpads,
                                                     const cryptonight::SplitContext* contexts,
                                                     unsigned long long* cycles)
{
    constexpr std::uint8_t VariantIndex = 2U;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    constexpr std::uint32_t sample_stride = 1024U;

    const unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int group_base = warp_lane & ~3U;
    const unsigned int group = global_tid >> 2;
    if (group >= count) return;
    const unsigned int mask = 0xFU << group_base;
    const bool profile_lane = group == 0U && lane == 0U;

    std::uint8_t* scratchpad = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const cryptonight::SplitContext& ctx = contexts[group];
    alignas(16) std::uint8_t a[16]{}, b[16]{}, c[16]{}, tbuf[16]{};
    std::uint64_t tweak = 0;
    if (lane == 0U) {
        cryptonight::copy16(a, ctx.a);
        cryptonight::copy16(b, ctx.b);
        tweak = ctx.tweak;
    }

    unsigned long long aes_load_cycles = 0ULL;
    unsigned long long write_mutate_cycles = 0ULL;
    unsigned long long dep_mul_cycles = 0ULL;
    unsigned long long sync_cycles = 0ULL;
    unsigned long long samples = 0ULL;

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const bool sampled = (iter & (sample_stride - 1U)) == 0U;
        unsigned long long t0 = 0ULL, t1 = 0ULL, t2 = 0ULL, t3 = 0ULL, t4 = 0ULL;
        if (profile_lane && sampled) t0 = clock64();

        std::uint32_t aw0=0,aw1=0,aw2=0,aw3=0,slot_index=0;
        if (lane == 0U) {
            aw0=cryptonight::load32le_aligned(a);
            aw1=cryptonight::load32le_aligned(a+4);
            aw2=cryptonight::load32le_aligned(a+8);
            aw3=cryptonight::load32le_aligned(a+12);
            slot_index=static_cast<std::uint32_t>(cryptonight::cn_index_masked(a,address_mask));
        }
        aw0=__shfl_sync(mask,aw0,group_base);
        aw1=__shfl_sync(mask,aw1,group_base+1U);
        aw2=__shfl_sync(mask,aw2,group_base+2U);
        aw3=__shfl_sync(mask,aw3,group_base+3U);
        slot_index=__shfl_sync(mask,slot_index,group_base);
        std::uint8_t* slot=scratchpad+static_cast<std::size_t>(slot_index)*16U;
        std::uint8_t x0,x1,x2,x3;
        if(lane==0U){x0=slot[0];x1=slot[5];x2=slot[10];x3=slot[15];}
        else if(lane==1U){x0=slot[4];x1=slot[9];x2=slot[14];x3=slot[3];}
        else if(lane==2U){x0=slot[8];x1=slot[13];x2=slot[2];x3=slot[7];}
        else{x0=slot[12];x1=slot[1];x2=slot[6];x3=slot[11];}
        const std::uint32_t key=lane==0U?aw0:lane==1U?aw1:lane==2U?aw2:aw3;
        const std::uint32_t cw=coop_mix_column(
            cryptonight::aes_sbox(x0),cryptonight::aes_sbox(x1),
            cryptonight::aes_sbox(x2),cryptonight::aes_sbox(x3))^key;
        const std::uint32_t c0=__shfl_sync(mask,cw,group_base);
        const std::uint32_t c1=__shfl_sync(mask,cw,group_base+1U);
        const std::uint32_t c2=__shfl_sync(mask,cw,group_base+2U);
        const std::uint32_t c3=__shfl_sync(mask,cw,group_base+3U);
        if (profile_lane && sampled) t1 = clock64();

        if(lane==0U){
            cryptonight::store32le_aligned(c,c0);
            cryptonight::store32le_aligned(c+4,c1);
            cryptonight::store32le_aligned(c+8,c2);
            cryptonight::store32le_aligned(c+12,c3);
            reinterpret_cast<std::uint64_t*>(slot)[0]=reinterpret_cast<const std::uint64_t*>(c)[0]^reinterpret_cast<const std::uint64_t*>(b)[0];
            reinterpret_cast<std::uint64_t*>(slot)[1]=reinterpret_cast<const std::uint64_t*>(c)[1]^reinterpret_cast<const std::uint64_t*>(b)[1];
            cryptonight::variant1_mutate(slot);
            if (profile_lane && sampled) t2 = clock64();

            slot=scratchpad+cryptonight::cn_index_masked(c,address_mask)*16U;
            cryptonight::copy16(tbuf,slot);
            const std::uint64_t cl=cryptonight::cn_load64_aligned(c);
            const std::uint64_t tv0=cryptonight::cn_load64_aligned(tbuf);
            const std::uint64_t lo=cl*tv0;
            const std::uint64_t hi=__umul64hi(cl,tv0);
            std::uint64_t a0=cryptonight::cn_load64_aligned(a)+hi;
            std::uint64_t a1=cryptonight::cn_load64_aligned(a+8)+lo;
            cryptonight::cn_store64_aligned(slot,a0);
            cryptonight::cn_store64_aligned(slot+8,a1^tweak);
            a0^=tv0;
            a1^=cryptonight::cn_load64_aligned(tbuf+8);
            cryptonight::cn_store64_aligned(a,a0);
            cryptonight::cn_store64_aligned(a+8,a1);
            cryptonight::copy16(b,c);
            if (profile_lane && sampled) t3 = clock64();
        }
        __syncwarp(mask);
        if (profile_lane && sampled) {
            t4 = clock64();
            aes_load_cycles += t1 - t0;
            write_mutate_cycles += t2 - t1;
            dep_mul_cycles += t3 - t2;
            sync_cycles += t4 - t3;
            ++samples;
        }
    }

    if (profile_lane) {
        cycles[0] = aes_load_cycles;
        cycles[1] = write_mutate_cycles;
        cycles[2] = dep_mul_cycles;
        cycles[3] = sync_cycles;
        cycles[4] = samples;
    }
}

void profile_cn_fast_coop4_inner(int device_id,
                                 cudaStream_t stream,
                                 std::uint8_t* states,
                                 std::size_t count,
                                 std::uint8_t* scratch,
                                 cryptonight::SplitContext* ctx,
                                 const CnGeometry& g)
{
    const char* benchmark = std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if (!benchmark || !*benchmark || std::string(benchmark) == "0" || count == 0U) return;

    unsigned long long* d_cycles = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cycles), 5U * sizeof(unsigned long long)),
               "CN-Fast inner profile alloc failed");
    try {
        check_cuda(cudaMemsetAsync(d_cycles, 0, 5U * sizeof(unsigned long long), stream),
                   "CN-Fast inner profile reset failed");
        check_cuda(cudaMemsetAsync(states, 0x6d, count * kStateBytes, stream),
                   "CN-Fast inner profile states failed");
        const int sb = static_cast<int>((count + static_cast<std::size_t>(g.setup_threads) - 1U) /
                                        static_cast<std::size_t>(g.setup_threads));
        if (g.aes_backend == 1)
            cryptonight_setup_stage<2, true><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        else
            cryptonight_setup_stage<2, false><<<sb, g.setup_threads, 0, stream>>>(states, count, scratch, ctx);
        check_cuda(cudaGetLastError(), "CN-Fast inner profile setup failed");

        int threads = 32;
        if (device_id >= 0 && device_id < kMaxLoopTuneDevices && g_cn_coop_ready[device_id]) {
            threads = g_cn_coop_threads[device_id][2];
            if (threads <= 0) threads = 32;
        }
        const int blocks = static_cast<int>((count * 4U + static_cast<std::size_t>(threads) - 1U) /
                                            static_cast<std::size_t>(threads));
        const float total_ms = time_cuda_phase(stream, [&]() {
            cryptonight_cn_fast_coop4_v1_profile<<<blocks, threads, 0, stream>>>(count, scratch, ctx, d_cycles);
            check_cuda(cudaGetLastError(), "CN-Fast inner profile launch failed");
        });

        std::array<unsigned long long, 5> h{};
        check_cuda(cudaMemcpyAsync(h.data(), d_cycles, sizeof(h), cudaMemcpyDeviceToHost, stream),
                   "CN-Fast inner profile copy failed");
        check_cuda(cudaStreamSynchronize(stream), "CN-Fast inner profile sync failed");
        cudaFree(d_cycles);
        d_cycles = nullptr;

        const double measured = static_cast<double>(h[0] + h[1] + h[2] + h[3]);
        auto pct = [&](unsigned long long v) {
            return measured > 0.0 ? 100.0 * static_cast<double>(v) / measured : 0.0;
        };
        std::cout << std::fixed << std::setprecision(2)
                  << "[CUDA CN-Fast inner profile] GPU " << device_id
                  << " | threads=" << threads
                  << " | samples=" << h[4]
                  << " | profiled_run=" << total_ms << " ms"
                  << " | load+AES=" << pct(h[0]) << "%"
                  << " | write+mutate=" << pct(h[1]) << "%"
                  << " | dep-load+mul+writeback=" << pct(h[2]) << "%"
                  << " | sync/state-exchange=" << pct(h[3]) << "%"
                  << std::defaultfloat << '\n';
    } catch (...) {
        if (d_cycles) cudaFree(d_cycles);
        throw;
    }
}

} // anonymous namespace
