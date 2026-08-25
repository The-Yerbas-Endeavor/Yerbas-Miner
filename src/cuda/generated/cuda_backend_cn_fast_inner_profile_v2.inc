// Corrected sampled inner-loop profiler for the established coop4-v1 CN-Fast
// kernel. All four A words are owned by lane 0 in v1, so every initial A-word
// broadcast comes from group_base exactly as in production.

namespace {

__global__ void cryptonight_cn_fast_coop4_v1_profile_v2(std::size_t count,
                                                        std::uint8_t* scratchpads,
                                                        const cryptonight::SplitContext* contexts,
                                                        unsigned long long* cycles)
{
    constexpr auto cfg = cryptonight::config_value(2);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    constexpr std::uint32_t sample_stride = 1024U;
    const unsigned int gt=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x&3U,wl=threadIdx.x&31U;
    const unsigned int base=wl&~3U,group=gt>>2;
    if(group>=count)return;
    const unsigned int mask=0xFU<<base;
    const bool prof=group==0U&&lane==0U;
    std::uint8_t* scratch=scratchpads+static_cast<std::size_t>(group)*cfg.page_size;
    const auto& ctx=contexts[group];
    alignas(16) std::uint8_t a[16]{},b[16]{},c[16]{},tbuf[16]{};
    std::uint64_t tweak=0;
    if(lane==0U){cryptonight::copy16(a,ctx.a);cryptonight::copy16(b,ctx.b);tweak=ctx.tweak;}
    unsigned long long s0=0,s1=0,s2=0,s3=0,ns=0;
    #pragma unroll 1
    for(std::uint32_t iter=0;iter<cfg.iterations;++iter){
        const bool sampled=(iter&(sample_stride-1U))==0U;
        unsigned long long t0=0,t1=0,t2=0,t3=0,t4=0;
        if(prof&&sampled)t0=clock64();
        std::uint32_t aw0=0,aw1=0,aw2=0,aw3=0,index=0;
        if(lane==0U){
            aw0=cryptonight::load32le_aligned(a);aw1=cryptonight::load32le_aligned(a+4);
            aw2=cryptonight::load32le_aligned(a+8);aw3=cryptonight::load32le_aligned(a+12);
            index=static_cast<std::uint32_t>(cryptonight::cn_index_masked(a,address_mask));
        }
        aw0=__shfl_sync(mask,aw0,base);aw1=__shfl_sync(mask,aw1,base);
        aw2=__shfl_sync(mask,aw2,base);aw3=__shfl_sync(mask,aw3,base);
        index=__shfl_sync(mask,index,base);
        std::uint8_t* slot=scratch+static_cast<std::size_t>(index)*16U;
        std::uint8_t x0,x1,x2,x3;
        if(lane==0U){x0=slot[0];x1=slot[5];x2=slot[10];x3=slot[15];}
        else if(lane==1U){x0=slot[4];x1=slot[9];x2=slot[14];x3=slot[3];}
        else if(lane==2U){x0=slot[8];x1=slot[13];x2=slot[2];x3=slot[7];}
        else{x0=slot[12];x1=slot[1];x2=slot[6];x3=slot[11];}
        const std::uint32_t key=lane==0U?aw0:lane==1U?aw1:lane==2U?aw2:aw3;
        const std::uint32_t cw=coop_mix_column(cryptonight::aes_sbox(x0),cryptonight::aes_sbox(x1),cryptonight::aes_sbox(x2),cryptonight::aes_sbox(x3))^key;
        const std::uint32_t c0=__shfl_sync(mask,cw,base),c1=__shfl_sync(mask,cw,base+1U);
        const std::uint32_t c2=__shfl_sync(mask,cw,base+2U),c3=__shfl_sync(mask,cw,base+3U);
        if(prof&&sampled)t1=clock64();
        if(lane==0U){
            cryptonight::store32le_aligned(c,c0);cryptonight::store32le_aligned(c+4,c1);
            cryptonight::store32le_aligned(c+8,c2);cryptonight::store32le_aligned(c+12,c3);
            reinterpret_cast<std::uint64_t*>(slot)[0]=reinterpret_cast<const std::uint64_t*>(c)[0]^reinterpret_cast<const std::uint64_t*>(b)[0];
            reinterpret_cast<std::uint64_t*>(slot)[1]=reinterpret_cast<const std::uint64_t*>(c)[1]^reinterpret_cast<const std::uint64_t*>(b)[1];
            cryptonight::variant1_mutate(slot);
            if(prof&&sampled)t2=clock64();
            slot=scratch+cryptonight::cn_index_masked(c,address_mask)*16U;
            cryptonight::copy16(tbuf,slot);
            const std::uint64_t cl=cryptonight::cn_load64_aligned(c),tv0=cryptonight::cn_load64_aligned(tbuf);
            const std::uint64_t lo=cl*tv0,hi=__umul64hi(cl,tv0);
            std::uint64_t a0=cryptonight::cn_load64_aligned(a)+hi,a1=cryptonight::cn_load64_aligned(a+8)+lo;
            cryptonight::cn_store64_aligned(slot,a0);cryptonight::cn_store64_aligned(slot+8,a1^tweak);
            a0^=tv0;a1^=cryptonight::cn_load64_aligned(tbuf+8);
            cryptonight::cn_store64_aligned(a,a0);cryptonight::cn_store64_aligned(a+8,a1);cryptonight::copy16(b,c);
            if(prof&&sampled)t3=clock64();
        }
        __syncwarp(mask);
        if(prof&&sampled){t4=clock64();s0+=t1-t0;s1+=t2-t1;s2+=t3-t2;s3+=t4-t3;++ns;}
    }
    if(prof){cycles[0]=s0;cycles[1]=s1;cycles[2]=s2;cycles[3]=s3;cycles[4]=ns;}
}

void profile_cn_fast_coop4_inner_v2(int device_id,cudaStream_t stream,std::uint8_t* states,
                                    std::size_t count,std::uint8_t* scratch,
                                    cryptonight::SplitContext* ctx,const CnGeometry& g)
{
    const char* benchmark=std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if(!benchmark||!*benchmark||std::string(benchmark)=="0"||count==0U)return;
    unsigned long long* d=nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d),5U*sizeof(unsigned long long)),"CN-Fast inner profile alloc failed");
    try{
        check_cuda(cudaMemsetAsync(d,0,5U*sizeof(unsigned long long),stream),"CN-Fast inner profile reset failed");
        check_cuda(cudaMemsetAsync(states,0x6d,count*kStateBytes,stream),"CN-Fast inner profile states failed");
        const int sb=static_cast<int>((count+static_cast<std::size_t>(g.setup_threads)-1U)/static_cast<std::size_t>(g.setup_threads));
        if(g.aes_backend==1)cryptonight_setup_stage<2,true><<<sb,g.setup_threads,0,stream>>>(states,count,scratch,ctx);
        else cryptonight_setup_stage<2,false><<<sb,g.setup_threads,0,stream>>>(states,count,scratch,ctx);
        check_cuda(cudaGetLastError(),"CN-Fast inner profile setup failed");
        int threads=32;
        if(device_id>=0&&device_id<kMaxLoopTuneDevices&&g_cn_coop_ready[device_id]){threads=g_cn_coop_threads[device_id][2];if(threads<=0)threads=32;}
        const int blocks=static_cast<int>((count*4U+static_cast<std::size_t>(threads)-1U)/static_cast<std::size_t>(threads));
        const float ms=time_cuda_phase(stream,[&](){cryptonight_cn_fast_coop4_v1_profile_v2<<<blocks,threads,0,stream>>>(count,scratch,ctx,d);check_cuda(cudaGetLastError(),"CN-Fast inner profile launch failed");});
        std::array<unsigned long long,5> h{};
        check_cuda(cudaMemcpyAsync(h.data(),d,sizeof(h),cudaMemcpyDeviceToHost,stream),"CN-Fast inner profile copy failed");
        check_cuda(cudaStreamSynchronize(stream),"CN-Fast inner profile sync failed");cudaFree(d);d=nullptr;
        const double total=static_cast<double>(h[0]+h[1]+h[2]+h[3]);
        auto pct=[&](unsigned long long v){return total>0.0?100.0*static_cast<double>(v)/total:0.0;};
        std::cout<<std::fixed<<std::setprecision(2)<<"[CUDA CN-Fast inner profile] GPU "<<device_id
                 <<" | threads="<<threads<<" | samples="<<h[4]<<" | profiled_run="<<ms<<" ms"
                 <<" | load+AES="<<pct(h[0])<<"% | write+mutate="<<pct(h[1])
                 <<"% | dep-load+mul+writeback="<<pct(h[2])<<"% | sync/state-exchange="<<pct(h[3])<<"%"
                 <<std::defaultfloat<<'\n';
    }catch(...){if(d)cudaFree(d);throw;}
}

} // anonymous namespace
