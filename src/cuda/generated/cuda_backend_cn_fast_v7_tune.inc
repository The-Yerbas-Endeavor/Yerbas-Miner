// Targeted generic CN-Fast v7 selector.
// v7 replaces constant-memory S-box + arithmetic MixColumns in coop4 with
// per-block shared-memory AES T-tables. Exact parity and >=2% loop improvement
// are required before production selection. No GPU model is hard-coded.

namespace {

constexpr int kCnFastV7CacheRevision = 1;
constexpr float kCnFastV7RequiredSpeedup = 0.98F;
static std::array<bool, kMaxLoopTuneDevices> g_cn_fast_v7_enabled{};
static std::array<bool, kMaxLoopTuneDevices> g_cn_fast_v7_ready{};

std::filesystem::path cn_fast_v7_cache_path(int device_id,const cudaDeviceProp& props,std::size_t batch_size)
{
    int dv=0,rv=0;
    if(cudaDriverGetVersion(&dv)!=cudaSuccess||cudaRuntimeGetVersion(&rv)!=cudaSuccess){cudaGetLastError();return {};}
    const std::string key=cn_cache_key(device_id,props,batch_size,dv,rv);
    return cn_cache_directory()/("cn-fast-v7-rev"+std::to_string(kCnFastV7CacheRevision)+'-'+key+".txt");
}

bool load_cn_fast_v7_cache(int device_id,const cudaDeviceProp& props,std::size_t batch_size)
{
    if(device_id<0||device_id>=kMaxLoopTuneDevices)return false;
    const char* r=std::getenv("YERBAS_CUDA_RETUNE"); const char* b=std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if((r&&*r&&std::string(r)!="0")||(b&&*b&&std::string(b)!="0"))return false;
    std::ifstream in(cn_fast_v7_cache_path(device_id,props,batch_size)); if(!in)return false;
    std::string magic; int rev=0,en=0;
    if(!(in>>magic>>rev>>en)||magic!="YERBAS_CN_FAST_V7"||rev!=kCnFastV7CacheRevision||(en!=0&&en!=1))return false;
    g_cn_fast_v7_enabled[device_id]=en!=0; g_cn_fast_v7_ready[device_id]=true;
    std::cout<<"[GPU "<<device_id<<"] CN-Fast v7 cache | selected="<<(en?"coop4-v7":"existing")<<'\n';
    return true;
}

void save_cn_fast_v7_cache(int device_id,const cudaDeviceProp& props,std::size_t batch_size)
{
    try{const auto p=cn_fast_v7_cache_path(device_id,props,batch_size);if(p.empty())return;std::filesystem::create_directories(p.parent_path());std::ofstream out(p,std::ios::trunc);if(out)out<<"YERBAS_CN_FAST_V7 "<<kCnFastV7CacheRevision<<' '<<(g_cn_fast_v7_enabled[device_id]?1:0)<<'\n';}catch(...){}
}

void launch_cn_fast_v7(cudaStream_t stream,int threads,std::size_t count,std::uint8_t* scratch,cryptonight::SplitContext* ctx)
{
    const int blocks=static_cast<int>((count*4U+static_cast<std::size_t>(threads)-1U)/static_cast<std::size_t>(threads));
    constexpr std::size_t shared_bytes=4U*256U*sizeof(std::uint32_t);
    cryptonight_loop_stage_coop4_v7<2><<<blocks,threads,shared_bytes,stream>>>(count,scratch,ctx);
}

void launch_cn_fast_existing_for_v7(int device_id,cudaStream_t stream,int threads,std::size_t count,std::uint8_t* scratch,cryptonight::SplitContext* ctx)
{
    if(device_id>=0&&device_id<kMaxLoopTuneDevices){
        if(g_cn_fast_v6_ready[device_id]&&g_cn_fast_v6_enabled[device_id]){launch_cn_fast_v6(stream,threads,count,scratch,ctx);return;}
        if(g_cn_fast_v5_ready[device_id]&&g_cn_fast_v5_enabled[device_id]){launch_cn_fast_v5(stream,threads,count,scratch,ctx);return;}
        if(g_cn_fast_v4_ready[device_id]&&g_cn_fast_v4_enabled[device_id]){launch_cn_fast_v4(stream,threads,count,scratch,ctx);return;}
    }
    int impl=1;if(device_id>=0&&device_id<kMaxLoopTuneDevices&&g_cn_coop_impl_ready[device_id])impl=g_cn_coop_impl[device_id][2];
    launch_cn_coop_impl<2>(stream,impl,threads,count,scratch,ctx);
}

bool validate_cn_fast_v7(int device_id,cudaStream_t stream,std::uint8_t* states,std::uint8_t* scratch,cryptonight::SplitContext* ctx,const CnGeometry& g,int threads)
{
    constexpr std::size_t n=2U; std::array<std::uint8_t,n*kStateBytes> a{},b{};
    auto run=[&](bool v7,auto& out){
        check_cuda(cudaMemsetAsync(states,0xa7,n*kStateBytes,stream),"CN-Fast v7 parity reset failed");
        const int sb=static_cast<int>((n+g.setup_threads-1U)/g.setup_threads);
        if(g.aes_backend==1)cryptonight_setup_stage<2,true><<<sb,g.setup_threads,0,stream>>>(states,n,scratch,ctx);else cryptonight_setup_stage<2,false><<<sb,g.setup_threads,0,stream>>>(states,n,scratch,ctx);
        check_cuda(cudaGetLastError(),"CN-Fast v7 parity setup failed");
        if(v7)launch_cn_fast_v7(stream,threads,n,scratch,ctx);else launch_cn_fast_existing_for_v7(device_id,stream,threads,n,scratch,ctx);
        check_cuda(cudaGetLastError(),"CN-Fast v7 parity loop failed");
        const int fb=static_cast<int>((n+g.final_threads-1U)/g.final_threads);
        if(g.aes_backend==1)cryptonight_final_stage<2,true><<<fb,g.final_threads,0,stream>>>(states,n,scratch,ctx);else cryptonight_final_stage<2,false><<<fb,g.final_threads,0,stream>>>(states,n,scratch,ctx);
        check_cuda(cudaGetLastError(),"CN-Fast v7 parity final failed");
        check_cuda(cudaMemcpyAsync(out.data(),states,out.size(),cudaMemcpyDeviceToHost,stream),"CN-Fast v7 parity copy failed");check_cuda(cudaStreamSynchronize(stream),"CN-Fast v7 parity sync failed");
    };
    run(false,a);run(true,b);return a==b;
}

void ensure_cn_fast_v7_mode(int device_id,const cudaDeviceProp& props,cudaStream_t stream,std::uint8_t* states,std::size_t count,std::uint8_t* scratch,cryptonight::SplitContext* ctx,const CnGeometry& g)
{
    if(device_id<0||device_id>=kMaxLoopTuneDevices||!g_cn_coop_enabled[device_id][2])return;
    if(load_cn_fast_v7_cache(device_id,props,count))return;
    g_cn_fast_v7_enabled[device_id]=false;g_cn_fast_v7_ready[device_id]=true;
    int threads=g_cn_coop_threads[device_id][2];if(threads<=0)threads=32;
    const bool parity=validate_cn_fast_v7(device_id,stream,states,scratch,ctx,g,threads);
    if(!parity){std::cout<<"[CUDA CN-Fast v7] parity=FAIL | selected=existing\n";save_cn_fast_v7_cache(device_id,props,count);return;}
    auto prep=[&](){check_cuda(cudaMemsetAsync(states,0xb3,count*kStateBytes,stream),"CN-Fast v7 benchmark reset failed");const int sb=static_cast<int>((count+g.setup_threads-1U)/g.setup_threads);if(g.aes_backend==1)cryptonight_setup_stage<2,true><<<sb,g.setup_threads,0,stream>>>(states,count,scratch,ctx);else cryptonight_setup_stage<2,false><<<sb,g.setup_threads,0,stream>>>(states,count,scratch,ctx);check_cuda(cudaGetLastError(),"CN-Fast v7 benchmark setup failed");};
    prep();const float base=time_cuda_phase(stream,[&](){launch_cn_fast_existing_for_v7(device_id,stream,threads,count,scratch,ctx);check_cuda(cudaGetLastError(),"CN-Fast existing-v7 benchmark failed");});
    prep();const float v7=time_cuda_phase(stream,[&](){launch_cn_fast_v7(stream,threads,count,scratch,ctx);check_cuda(cudaGetLastError(),"CN-Fast v7 benchmark failed");});
    const float gain=(base/v7-1.0F)*100.0F;const bool use=v7<base*kCnFastV7RequiredSpeedup;g_cn_fast_v7_enabled[device_id]=use;
    std::cout<<std::fixed<<std::setprecision(3)<<"[CUDA CN-Fast v7] GPU "<<device_id<<" | threads="<<threads<<" | baseline="<<base<<" ms | v7="<<v7<<" ms | parity=PASS | gain="<<gain<<"% | selected="<<(use?"coop4-v7":"existing")<<std::defaultfloat<<'\n';
    save_cn_fast_v7_cache(device_id,props,count);
}

} // anonymous namespace
