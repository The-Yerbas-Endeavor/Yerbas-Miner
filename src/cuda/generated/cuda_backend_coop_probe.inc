// Experimental cooperative CryptoNight loop probes.
//
// Production dispatch remains unchanged.  YERBAS_CUDA_COOP_PROBE=1 parity-checks
// and benchmarks the known-good single-thread loop against two 4-lane-per-hash
// implementations.  v2 follows the useful part of the KAWPOW CUDA model: keep
// state in lane-owned registers and exchange only the dependent values through
// sub-warp shuffles instead of rebuilding 16-byte temporary arrays in lane 0.

namespace {

__device__ __forceinline__ std::uint32_t coop_mix_column(std::uint8_t a0, std::uint8_t a1,
                                                         std::uint8_t a2, std::uint8_t a3)
{
    const std::uint8_t t = static_cast<std::uint8_t>(a0 ^ a1 ^ a2 ^ a3);
    const std::uint8_t o0 = static_cast<std::uint8_t>(a0 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a0 ^ a1)));
    const std::uint8_t o1 = static_cast<std::uint8_t>(a1 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a1 ^ a2)));
    const std::uint8_t o2 = static_cast<std::uint8_t>(a2 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a2 ^ a3)));
    const std::uint8_t o3 = static_cast<std::uint8_t>(a3 ^ t ^ cryptonight::xtime(static_cast<std::uint8_t>(a3 ^ a0)));
    return cryptonight::pack4(o0, o1, o2, o3);
}

// First-generation probe retained as a baseline.
template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_coop4(std::size_t count, std::uint8_t* scratchpads,
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
    alignas(16) std::uint8_t a[16]{}, b[16]{}, c[16]{}, tbuf[16]{};
    std::uint64_t tweak = 0;
    if (lane == 0U) { cryptonight::copy16(a, ctx.a); cryptonight::copy16(b, ctx.b); tweak = ctx.tweak; }
    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        std::uint32_t aw0=0,aw1=0,aw2=0,aw3=0,slot_index=0;
        if (lane == 0U) {
            aw0=cryptonight::load32le_aligned(a); aw1=cryptonight::load32le_aligned(a+4);
            aw2=cryptonight::load32le_aligned(a+8); aw3=cryptonight::load32le_aligned(a+12);
            slot_index=static_cast<std::uint32_t>(cryptonight::cn_index_masked(a,address_mask));
        }
        aw0=__shfl_sync(mask,aw0,group_base); aw1=__shfl_sync(mask,aw1,group_base);
        aw2=__shfl_sync(mask,aw2,group_base); aw3=__shfl_sync(mask,aw3,group_base);
        slot_index=__shfl_sync(mask,slot_index,group_base);
        std::uint8_t* slot=scratchpad+static_cast<std::size_t>(slot_index)*16U;
        std::uint8_t x0,x1,x2,x3;
        if(lane==0U){x0=slot[0];x1=slot[5];x2=slot[10];x3=slot[15];}
        else if(lane==1U){x0=slot[4];x1=slot[9];x2=slot[14];x3=slot[3];}
        else if(lane==2U){x0=slot[8];x1=slot[13];x2=slot[2];x3=slot[7];}
        else{x0=slot[12];x1=slot[1];x2=slot[6];x3=slot[11];}
        const std::uint32_t key=lane==0U?aw0:lane==1U?aw1:lane==2U?aw2:aw3;
        const std::uint32_t cw=coop_mix_column(cryptonight::aes_sbox(x0),cryptonight::aes_sbox(x1),cryptonight::aes_sbox(x2),cryptonight::aes_sbox(x3))^key;
        const std::uint32_t c0=__shfl_sync(mask,cw,group_base), c1=__shfl_sync(mask,cw,group_base+1U);
        const std::uint32_t c2=__shfl_sync(mask,cw,group_base+2U), c3=__shfl_sync(mask,cw,group_base+3U);
        if(lane==0U){
            cryptonight::store32le_aligned(c,c0);cryptonight::store32le_aligned(c+4,c1);cryptonight::store32le_aligned(c+8,c2);cryptonight::store32le_aligned(c+12,c3);
            reinterpret_cast<std::uint64_t*>(slot)[0]=reinterpret_cast<const std::uint64_t*>(c)[0]^reinterpret_cast<const std::uint64_t*>(b)[0];
            reinterpret_cast<std::uint64_t*>(slot)[1]=reinterpret_cast<const std::uint64_t*>(c)[1]^reinterpret_cast<const std::uint64_t*>(b)[1];
            cryptonight::variant1_mutate(slot);
            slot=scratchpad+cryptonight::cn_index_masked(c,address_mask)*16U; cryptonight::copy16(tbuf,slot);
            const std::uint64_t cl=cryptonight::cn_load64_aligned(c), t0=cryptonight::cn_load64_aligned(tbuf);
            const std::uint64_t lo=cl*t0, hi=__umul64hi(cl,t0);
            std::uint64_t a0=cryptonight::cn_load64_aligned(a)+hi,a1=cryptonight::cn_load64_aligned(a+8)+lo;
            cryptonight::cn_store64_aligned(slot,a0);cryptonight::cn_store64_aligned(slot+8,a1^tweak);
            a0^=t0;a1^=cryptonight::cn_load64_aligned(tbuf+8);cryptonight::cn_store64_aligned(a,a0);cryptonight::cn_store64_aligned(a+8,a1);cryptonight::copy16(b,c);
        }
        __syncwarp(mask);
    }
}

// v2 keeps A/B/C distributed as one 32-bit word per lane.  The dependent
// multiply remains in lane 0, but every subwarp collective is executed by all
// four participating lanes and the complete updated A state is handed back.
template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_coop4_v2(std::size_t count, std::uint8_t* scratchpads,
                                                const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg=cryptonight::config_value(VariantIndex); constexpr std::size_t address_mask=cfg.aes_rounds-1U;
    const unsigned int gt=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x&3U,wl=threadIdx.x&31U,base=wl&~3U,group=gt>>2;
    if(group>=count)return; const unsigned int mask=0xFU<<base;
    std::uint8_t* scratch=scratchpads+static_cast<std::size_t>(group)*cfg.page_size; const auto& ctx=contexts[group];
    std::uint32_t aw=cryptonight::load32le_aligned(ctx.a+lane*4U), bw=cryptonight::load32le_aligned(ctx.b+lane*4U);
    const std::uint64_t tweak=ctx.tweak;
    #pragma unroll 1
    for(std::uint32_t iter=0;iter<cfg.iterations;++iter){
        const std::uint32_t aw0=__shfl_sync(mask,aw,base);
        const std::uint32_t index=aw0&static_cast<std::uint32_t>(address_mask);
        std::uint8_t* slot=scratch+static_cast<std::size_t>(index)*16U;
        std::uint8_t x0,x1,x2,x3;
        if(lane==0U){x0=slot[0];x1=slot[5];x2=slot[10];x3=slot[15];}
        else if(lane==1U){x0=slot[4];x1=slot[9];x2=slot[14];x3=slot[3];}
        else if(lane==2U){x0=slot[8];x1=slot[13];x2=slot[2];x3=slot[7];}
        else{x0=slot[12];x1=slot[1];x2=slot[6];x3=slot[11];}
        const std::uint32_t cw=coop_mix_column(cryptonight::aes_sbox(x0),cryptonight::aes_sbox(x1),cryptonight::aes_sbox(x2),cryptonight::aes_sbox(x3))^aw;
        reinterpret_cast<std::uint32_t*>(slot)[lane]=cw^bw;
        __syncwarp(mask);
        if(lane==0U)cryptonight::variant1_mutate(slot);

        const std::uint32_t c0=__shfl_sync(mask,cw,base), c1=__shfl_sync(mask,cw,base+1U);
        const std::uint32_t c2=__shfl_sync(mask,cw,base+2U), c3=__shfl_sync(mask,cw,base+3U);
        const std::uint32_t old_a0=__shfl_sync(mask,aw,base), old_a1=__shfl_sync(mask,aw,base+1U);
        const std::uint32_t old_a2=__shfl_sync(mask,aw,base+2U), old_a3=__shfl_sync(mask,aw,base+3U);

        std::uint32_t next_a0=0U,next_a1=0U,next_a2=0U,next_a3=0U;
        if(lane==0U){
            const std::uint32_t j=c0&static_cast<std::uint32_t>(address_mask); std::uint8_t* slot2=scratch+static_cast<std::size_t>(j)*16U;
            const std::uint64_t cl=static_cast<std::uint64_t>(c0)|(static_cast<std::uint64_t>(c1)<<32U);
            const std::uint64_t t0=cryptonight::cn_load64_aligned(slot2),t1=cryptonight::cn_load64_aligned(slot2+8);
            const std::uint64_t lo=cl*t0,hi=__umul64hi(cl,t0);
            const std::uint64_t olda0=static_cast<std::uint64_t>(old_a0)|(static_cast<std::uint64_t>(old_a1)<<32U);
            const std::uint64_t olda1=static_cast<std::uint64_t>(old_a2)|(static_cast<std::uint64_t>(old_a3)<<32U);
            const std::uint64_t na0=olda0+hi,na1=olda1+lo;
            cryptonight::cn_store64_aligned(slot2,na0);cryptonight::cn_store64_aligned(slot2+8,na1^tweak);
            const std::uint64_t xa0=na0^t0,xa1=na1^t1;
            next_a0=static_cast<std::uint32_t>(xa0);
            next_a1=static_cast<std::uint32_t>(xa0>>32U);
            next_a2=static_cast<std::uint32_t>(xa1);
            next_a3=static_cast<std::uint32_t>(xa1>>32U);
        }

        const std::uint32_t out_a0=__shfl_sync(mask,next_a0,base);
        const std::uint32_t out_a1=__shfl_sync(mask,next_a1,base);
        const std::uint32_t out_a2=__shfl_sync(mask,next_a2,base);
        const std::uint32_t out_a3=__shfl_sync(mask,next_a3,base);
        aw=lane==0U?out_a0:lane==1U?out_a1:lane==2U?out_a2:out_a3;
        bw=cw;
        __syncwarp(mask);
    }
}

__global__ void compare_bytes_kernel(const std::uint8_t* a,const std::uint8_t* b,std::size_t bytes,unsigned int* mismatches){const std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i<bytes&&a[i]!=b[i])atomicAdd(mismatches,1U);}

template <std::uint8_t V, typename Launch>
bool coop_parity(cudaStream_t stream,const CnGeometry& g,std::uint8_t* states,Launch launch){
    constexpr std::size_t sample=2; constexpr auto cfg=cryptonight::config_value(V); const std::size_t bytes=sample*static_cast<std::size_t>(cfg.page_size);
    std::uint8_t *rs=nullptr,*cs=nullptr; cryptonight::SplitContext *rc=nullptr,*cc=nullptr; unsigned int* mm=nullptr;
    try{check_cuda(cudaMalloc((void**)&rs,bytes),"coop ref alloc");check_cuda(cudaMalloc((void**)&cs,bytes),"coop test alloc");check_cuda(cudaMalloc((void**)&rc,sample*sizeof(*rc)),"coop ref ctx");check_cuda(cudaMalloc((void**)&cc,sample*sizeof(*cc)),"coop test ctx");check_cuda(cudaMalloc((void**)&mm,sizeof(*mm)),"coop mismatch alloc");
        check_cuda(cudaMemsetAsync(states,0x47U+V*19U,sample*kStateBytes,stream),"coop state prep"); const int st=g.setup_threads,sb=static_cast<int>((sample+st-1U)/st);
        cryptonight_setup_stage<V,false><<<sb,st,0,stream>>>(states,sample,rs,rc);cryptonight_setup_stage<V,false><<<sb,st,0,stream>>>(states,sample,cs,cc);check_cuda(cudaGetLastError(),"coop setup");
        launch_cn_loop_backend<V>(stream,1,false,1,32,sample,rs,rc);launch(sample,cs,cc);check_cuda(cudaGetLastError(),"coop launch");check_cuda(cudaMemsetAsync(mm,0,sizeof(*mm),stream),"coop reset");
        const int ct=256,cb=static_cast<int>((bytes+ct-1U)/ct);compare_bytes_kernel<<<cb,ct,0,stream>>>(rs,cs,bytes,mm);unsigned int hm=0;check_cuda(cudaMemcpyAsync(&hm,mm,sizeof(hm),cudaMemcpyDeviceToHost,stream),"coop copy");check_cuda(cudaStreamSynchronize(stream),"coop sync");
        cudaFree(rs);cudaFree(cs);cudaFree(rc);cudaFree(cc);cudaFree(mm);return hm==0U;
    }catch(...){if(rs)cudaFree(rs);if(cs)cudaFree(cs);if(rc)cudaFree(rc);if(cc)cudaFree(cc);if(mm)cudaFree(mm);throw;}}

template <std::uint8_t V>
void coop_benchmark_variant(cudaStream_t stream,std::uint8_t* states,std::size_t n,std::uint8_t* scratch,cryptonight::SplitContext* ctx,const CnGeometry& g){
    auto l1=[&](std::size_t c,std::uint8_t*s,cryptonight::SplitContext*x){cryptonight_loop_stage_coop4<V><<<1,32,0,stream>>>(c,s,x);};
    auto l2=[&](std::size_t c,std::uint8_t*s,cryptonight::SplitContext*x){cryptonight_loop_stage_coop4_v2<V><<<1,32,0,stream>>>(c,s,x);};
    const bool p1=coop_parity<V>(stream,g,states,l1),p2=coop_parity<V>(stream,g,states,l2);
    const int st=g.setup_threads,sb=static_cast<int>((n+st-1U)/st); auto prep=[&](){check_cuda(cudaMemsetAsync(states,0x63U+V*13U,n*kStateBytes,stream),"coop bench prep");cryptonight_setup_stage<V,false><<<sb,st,0,stream>>>(states,n,scratch,ctx);check_cuda(cudaGetLastError(),"coop bench setup");};
    prep();const int bt=g.loop_threads,bb=static_cast<int>((n+bt-1U)/bt);const float single=time_cuda_phase(stream,[&](){launch_cn_loop_backend<V>(stream,1,false,bb,bt,n,scratch,ctx);});
    float m1=0,m2=0;constexpr int threads=128;const int blocks=static_cast<int>((n*4U+threads-1U)/threads);
    if(p1){prep();m1=time_cuda_phase(stream,[&](){cryptonight_loop_stage_coop4<V><<<blocks,threads,0,stream>>>(n,scratch,ctx);});}
    if(p2){prep();m2=time_cuda_phase(stream,[&](){cryptonight_loop_stage_coop4_v2<V><<<blocks,threads,0,stream>>>(n,scratch,ctx);});}
    std::cout<<std::fixed<<std::setprecision(3)<<"[CUDA CN coop probe] "<<cryptonight::config_value(V).name<<" | hashes="<<n<<" | single="<<single<<" ms | coop4="<<(p1?m1:0.0F)<<" ms("<<(p1?"PASS":"FAIL")<<") | coop4-v2="<<(p2?m2:0.0F)<<" ms("<<(p2?"PASS":"FAIL")<<") | production=single"<<std::defaultfloat<<'\n';
}

} // anonymous namespace

std::array<CnGeometry,6> autotune_cn_geometries(int device_id,cudaStream_t stream,std::uint8_t* states,std::size_t batch_size,std::uint8_t* scratchpads,cryptonight::SplitContext* contexts){
    auto result=autotune_cn_geometries_prod(device_id,stream,states,batch_size,scratchpads,contexts);const char* probe=std::getenv("YERBAS_CUDA_COOP_PROBE");if(!(probe&&*probe&&std::string(probe)!="0"))return result;
    std::cout<<"[GPU "<<device_id<<"] experimental CryptoNight cooperative probe starting | hashes="<<batch_size<<"\n";
    coop_benchmark_variant<0>(stream,states,batch_size,scratchpads,contexts,result[0]);coop_benchmark_variant<1>(stream,states,batch_size,scratchpads,contexts,result[1]);coop_benchmark_variant<2>(stream,states,batch_size,scratchpads,contexts,result[2]);coop_benchmark_variant<3>(stream,states,batch_size,scratchpads,contexts,result[3]);coop_benchmark_variant<4>(stream,states,batch_size,scratchpads,contexts,result[4]);coop_benchmark_variant<5>(stream,states,batch_size,scratchpads,contexts,result[5]);
    std::cout<<"[GPU "<<device_id<<"] experimental CryptoNight cooperative probe complete | production unchanged\n";return result;
}
