// Runtime selector for register-word CryptoNight setup/final candidates.
// Architecture-generic: exact parity is mandatory and each phase is promoted
// independently only after a repeated median >=5% win.

namespace {

constexpr int kCnWordPhaseMaxDevices = 32;
constexpr int kCnWordPhasePasses = 3;
constexpr float kCnWordPhasePromotionRatio = 0.95F;
static std::array<std::array<int, 6>, kCnWordPhaseMaxDevices> g_cn_word_setup{};
static std::array<std::array<int, 6>, kCnWordPhaseMaxDevices> g_cn_word_final{};

template <std::size_t N>
float cn_wordphase_median(std::array<float, N> x)
{
    std::sort(x.begin(), x.end());
    return x[N / 2U];
}

template <std::uint8_t VariantIndex>
void cn_launch_setup_phase(cudaStream_t stream, bool word, std::size_t count,
                           std::uint8_t* states, std::uint8_t* scratchpads,
                           cryptonight::SplitContext* contexts)
{
    constexpr int threads = 128;
    const std::size_t total = count * 8U;
    const int blocks = static_cast<int>((total + threads - 1U) / threads);
    if (word)
        cryptonight_setup_stage_word8<VariantIndex><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_setup_stage_cooperative8_sharedkey<VariantIndex, true><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN setup selector launch failed");
}

template <std::uint8_t VariantIndex>
void cn_launch_final_phase(cudaStream_t stream, bool word, std::size_t count,
                           std::uint8_t* states, std::uint8_t* scratchpads,
                           cryptonight::SplitContext* contexts)
{
    constexpr int threads = 128;
    const std::size_t total = count * 8U;
    const int blocks = static_cast<int>((total + threads - 1U) / threads);
    if (word)
        cryptonight_final_stage_word8<VariantIndex><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
    else
        cryptonight_final_stage_cooperative8_sharedkey<VariantIndex, true><<<blocks, threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN final selector launch failed");
}

template <typename Launch>
float cn_wordphase_time(cudaStream_t stream, Launch&& launch)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate CN wordphase start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate CN wordphase stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN wordphase start failed");
    launch();
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN wordphase stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN wordphase failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime CN wordphase failed");
    cudaEventDestroy(stop); cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
bool cn_validate_setup_word(cudaStream_t stream, const std::uint8_t* states)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t *s0=nullptr,*s1=nullptr,*p0=nullptr,*p1=nullptr;
    cryptonight::SplitContext *c0=nullptr,*c1=nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s0), kStateBytes), "cudaMalloc setup parity state0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s1), kStateBytes), "cudaMalloc setup parity state1 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p0), cfg.page_size), "cudaMalloc setup parity pad0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p1), cfg.page_size), "cudaMalloc setup parity pad1 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c0), sizeof(*c0)), "cudaMalloc setup parity ctx0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c1), sizeof(*c1)), "cudaMalloc setup parity ctx1 failed");
    check_cuda(cudaMemcpyAsync(s0, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy setup parity state0 failed");
    check_cuda(cudaMemcpyAsync(s1, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy setup parity state1 failed");
    cn_launch_setup_phase<VariantIndex>(stream, false, 1, s0, p0, c0);
    cn_launch_setup_phase<VariantIndex>(stream, true, 1, s1, p1, c1);
    std::vector<std::uint8_t> hp0(cfg.page_size), hp1(cfg.page_size);
    cryptonight::SplitContext hc0{}, hc1{};
    check_cuda(cudaMemcpyAsync(hp0.data(), p0, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity pad0 result failed");
    check_cuda(cudaMemcpyAsync(hp1.data(), p1, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity pad1 result failed");
    check_cuda(cudaMemcpyAsync(&hc0, c0, sizeof(hc0), cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity ctx0 result failed");
    check_cuda(cudaMemcpyAsync(&hc1, c1, sizeof(hc1), cudaMemcpyDeviceToHost, stream), "cudaMemcpy setup parity ctx1 result failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize setup parity failed");
    cudaFree(c1); cudaFree(c0); cudaFree(p1); cudaFree(p0); cudaFree(s1); cudaFree(s0);
    return hp0 == hp1 && std::memcmp(&hc0, &hc1, sizeof(hc0)) == 0;
}

template <std::uint8_t VariantIndex>
bool cn_validate_final_word(cudaStream_t stream, const std::uint8_t* states,
                            const std::uint8_t* scratchpads,
                            const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t *s0=nullptr,*s1=nullptr,*p0=nullptr,*p1=nullptr;
    cryptonight::SplitContext *c0=nullptr,*c1=nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s0), kStateBytes), "cudaMalloc final parity state0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s1), kStateBytes), "cudaMalloc final parity state1 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p0), cfg.page_size), "cudaMalloc final parity pad0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p1), cfg.page_size), "cudaMalloc final parity pad1 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c0), sizeof(*c0)), "cudaMalloc final parity ctx0 failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c1), sizeof(*c1)), "cudaMalloc final parity ctx1 failed");
    check_cuda(cudaMemcpyAsync(s0, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity state0 failed");
    check_cuda(cudaMemcpyAsync(s1, states, kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity state1 failed");
    check_cuda(cudaMemcpyAsync(p0, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity pad0 failed");
    check_cuda(cudaMemcpyAsync(p1, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity pad1 failed");
    check_cuda(cudaMemcpyAsync(c0, contexts, sizeof(*c0), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity ctx0 failed");
    check_cuda(cudaMemcpyAsync(c1, contexts, sizeof(*c1), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy final parity ctx1 failed");
    cn_launch_final_phase<VariantIndex>(stream, false, 1, s0, p0, c0);
    cn_launch_final_phase<VariantIndex>(stream, true, 1, s1, p1, c1);
    std::array<std::uint8_t, kStateBytes> hs0{}, hs1{};
    cryptonight::SplitContext hc0{}, hc1{};
    check_cuda(cudaMemcpyAsync(hs0.data(), s0, kStateBytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity state0 result failed");
    check_cuda(cudaMemcpyAsync(hs1.data(), s1, kStateBytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity state1 result failed");
    check_cuda(cudaMemcpyAsync(&hc0, c0, sizeof(hc0), cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity ctx0 result failed");
    check_cuda(cudaMemcpyAsync(&hc1, c1, sizeof(hc1), cudaMemcpyDeviceToHost, stream), "cudaMemcpy final parity ctx1 result failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize final parity failed");
    cudaFree(c1); cudaFree(c0); cudaFree(p1); cudaFree(p0); cudaFree(s1); cudaFree(s0);
    return hs0 == hs1 && std::memcmp(&hc0, &hc1, sizeof(hc0)) == 0;
}

template <std::uint8_t VariantIndex>
void cn_select_setup_word(cudaStream_t stream, int device_id, std::size_t count,
                          std::uint8_t* states)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices || g_cn_word_setup[device_id][VariantIndex] != 0) return;
    const bool parity = cn_validate_setup_word<VariantIndex>(stream, states);
    if (!parity) { g_cn_word_setup[device_id][VariantIndex] = -1; return; }
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    const std::size_t n = std::min<std::size_t>(count, 4U);
    std::uint8_t *s=nullptr,*p=nullptr; cryptonight::SplitContext* c=nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s), n*kStateBytes), "cudaMalloc setup timing state failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p), n*cfg.page_size), "cudaMalloc setup timing pad failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c), n*sizeof(*c)), "cudaMalloc setup timing ctx failed");
    std::array<float,kCnWordPhasePasses> base{}, word{};
    for (int i=0;i<kCnWordPhasePasses;++i) {
        check_cuda(cudaMemcpyAsync(s, states, n*kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy setup timing base failed");
        base[i]=cn_wordphase_time(stream,[&]{cn_launch_setup_phase<VariantIndex>(stream,false,n,s,p,c);});
        check_cuda(cudaMemcpyAsync(s, states, n*kStateBytes, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy setup timing word failed");
        word[i]=cn_wordphase_time(stream,[&]{cn_launch_setup_phase<VariantIndex>(stream,true,n,s,p,c);});
    }
    cudaFree(c); cudaFree(p); cudaFree(s);
    const float b=cn_wordphase_median(base), w=cn_wordphase_median(word);
    g_cn_word_setup[device_id][VariantIndex]=(w < b*kCnWordPhasePromotionRatio)?1:-1;
    std::cout<<std::fixed<<std::setprecision(3)<<"[CUDA CN wordphase] GPU "<<device_id<<" | "<<cfg.name
             <<" | setup baseline="<<b<<" ms word="<<w<<" ms parity=PASS production="
             <<(g_cn_word_setup[device_id][VariantIndex]>0?"word32":"sharedkey")<<std::defaultfloat<<'\n';
}

template <std::uint8_t VariantIndex>
void cn_select_final_word(cudaStream_t stream, int device_id, std::size_t count,
                          std::uint8_t* states, std::uint8_t* scratchpads,
                          cryptonight::SplitContext* contexts)
{
    if (device_id < 0 || device_id >= kCnWordPhaseMaxDevices || g_cn_word_final[device_id][VariantIndex] != 0) return;
    const bool parity=cn_validate_final_word<VariantIndex>(stream,states,scratchpads,contexts);
    if (!parity) { g_cn_word_final[device_id][VariantIndex]=-1; return; }
    constexpr auto cfg=cryptonight::config_value(VariantIndex);
    const std::size_t n=std::min<std::size_t>(count,4U);
    std::uint8_t *s=nullptr,*p=nullptr; cryptonight::SplitContext* c=nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&s),n*kStateBytes),"cudaMalloc final timing state failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&p),n*cfg.page_size),"cudaMalloc final timing pad failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c),n*sizeof(*c)),"cudaMalloc final timing ctx failed");
    auto reset=[&]{
        check_cuda(cudaMemcpyAsync(s,states,n*kStateBytes,cudaMemcpyDeviceToDevice,stream),"cudaMemcpy final timing state failed");
        check_cuda(cudaMemcpyAsync(p,scratchpads,n*cfg.page_size,cudaMemcpyDeviceToDevice,stream),"cudaMemcpy final timing pad failed");
        check_cuda(cudaMemcpyAsync(c,contexts,n*sizeof(*c),cudaMemcpyDeviceToDevice,stream),"cudaMemcpy final timing ctx failed");
    };
    std::array<float,kCnWordPhasePasses> base{},word{};
    for(int i=0;i<kCnWordPhasePasses;++i){
        reset(); base[i]=cn_wordphase_time(stream,[&]{cn_launch_final_phase<VariantIndex>(stream,false,n,s,p,c);});
        reset(); word[i]=cn_wordphase_time(stream,[&]{cn_launch_final_phase<VariantIndex>(stream,true,n,s,p,c);});
    }
    cudaFree(c); cudaFree(p); cudaFree(s);
    const float b=cn_wordphase_median(base),w=cn_wordphase_median(word);
    g_cn_word_final[device_id][VariantIndex]=(w < b*kCnWordPhasePromotionRatio)?1:-1;
    std::cout<<std::fixed<<std::setprecision(3)<<"[CUDA CN wordphase] GPU "<<device_id<<" | "<<cfg.name
             <<" | final baseline="<<b<<" ms word="<<w<<" ms parity=PASS production="
             <<(g_cn_word_final[device_id][VariantIndex]>0?"word32":"sharedkey")<<std::defaultfloat<<'\n';
}

template <std::uint8_t VariantIndex>
void launch_cn_setup_selected(cudaStream_t stream,int device_id,std::size_t count,
                              std::uint8_t* states,std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts)
{
    cn_select_setup_word<VariantIndex>(stream,device_id,count,states);
    cn_launch_setup_phase<VariantIndex>(stream,g_cn_word_setup[device_id][VariantIndex]>0,count,states,scratchpads,contexts);
}

template <std::uint8_t VariantIndex>
void launch_cn_final_selected(cudaStream_t stream,int device_id,std::size_t count,
                              std::uint8_t* states,std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts)
{
    cn_select_final_word<VariantIndex>(stream,device_id,count,states,scratchpads,contexts);
    cn_launch_final_phase<VariantIndex>(stream,g_cn_word_final[device_id][VariantIndex]>0,count,states,scratchpads,contexts);
}

inline const char* cn_word_setup_name(int device_id,std::uint8_t variant)
{
    return (device_id>=0&&device_id<kCnWordPhaseMaxDevices&&variant<6&&g_cn_word_setup[device_id][variant]>0)?"word32":"sharedkey";
}
inline const char* cn_word_final_name(int device_id,std::uint8_t variant)
{
    return (device_id>=0&&device_id<kCnWordPhaseMaxDevices&&variant<6&&g_cn_word_final[device_id][variant]>0)?"word32":"sharedkey";
}

} // anonymous namespace
