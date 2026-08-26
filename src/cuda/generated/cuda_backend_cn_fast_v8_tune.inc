// Generic CN-Fast v8 occupancy selector.
//
// v8 keeps CryptoNight semantics unchanged and only compares two NVCC launch-
// bounds variants of the cooperative loop to test whether lower register pressure
// and higher resident-block count can hide the serialized memory latency.
// Exact parity and >=2% loop improvement are required. No GPU model is hard-coded.

namespace {

constexpr int kCnFastV8CacheRevision = 1;
constexpr float kCnFastV8RequiredSpeedup = 0.98F;
static std::array<int, kMaxLoopTuneDevices> g_cn_fast_v8_mode{}; // 0 existing, 2 b2, 4 b4
static std::array<bool, kMaxLoopTuneDevices> g_cn_fast_v8_ready{};

std::filesystem::path cn_fast_v8_cache_path(int device_id,const cudaDeviceProp& props,std::size_t batch_size)
{
    int dv=0,rv=0;
    if(cudaDriverGetVersion(&dv)!=cudaSuccess||cudaRuntimeGetVersion(&rv)!=cudaSuccess){cudaGetLastError();return {};}
    const std::string key=cn_cache_key(device_id,props,batch_size,dv,rv);
    return cn_cache_directory()/("cn-fast-v8-rev"+std::to_string(kCnFastV8CacheRevision)+'-'+key+".txt");
}

bool load_cn_fast_v8_cache(int device_id,const cudaDeviceProp& props,std::size_t batch_size)
{
    if(device_id<0||device_id>=kMaxLoopTuneDevices)return false;
    const char* r=std::getenv("YERBAS_CUDA_RETUNE"); const char* b=std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if((r&&*r&&std::string(r)!="0")||(b&&*b&&std::string(b)!="0"))return false;
    std::ifstream in(cn_fast_v8_cache_path(device_id,props,batch_size)); if(!in)return false;
    std::string magic; int rev=0,mode=0;
    if(!(in>>magic>>rev>>mode)||magic!="YERBAS_CN_FAST_V8"||rev!=kCnFastV8CacheRevision||
       (mode!=0&&mode!=2&&mode!=4))return false;
    g_cn_fast_v8_mode[device_id]=mode; g_cn_fast_v8_ready[device_id]=true;
    std::cout<<"[GPU "<<device_id<<"] CN-Fast v8 cache | selected="
             <<(mode==2?"coop4-v8-b2":mode==4?"coop4-v8-b4":"existing")<<'\n';
    return true;
}

void save_cn_fast_v8_cache(int device_id,const cudaDeviceProp& props,std::size_t batch_size)
{
    try{const auto p=cn_fast_v8_cache_path(device_id,props,batch_size);if(p.empty())return;std::filesystem::create_directories(p.parent_path());std::ofstream out(p,std::ios::trunc);if(out)out<<"YERBAS_CN_FAST_V8 "<<kCnFastV8CacheRevision<<' '<<g_cn_fast_v8_mode[device_id]<<'\n';}catch(...){}
}

void launch_cn_fast_v8(cudaStream_t stream,int mode,int threads,std::size_t count,std::uint8_t* scratch,cryptonight::SplitContext* ctx)
{
    const int blocks=static_cast<int>((count*4U+static_cast<std::size_t>(threads)-1U)/static_cast<std::size_t>(threads));
    if(mode==4) cryptonight_loop_stage_coop4_v8_b4<2><<<blocks,threads,0,stream>>>(count,scratch,ctx);
    else cryptonight_loop_stage_coop4_v8_b2<2><<<blocks,threads,0,stream>>>(count,scratch,ctx);
}

void launch_cn_fast_existing_for_v8(int device_id,cudaStream_t stream,int threads,std::size_t count,std::uint8_t* scratch,cryptonight::SplitContext* ctx)
{
    if(device_id>=0&&device_id<kMaxLoopTuneDevices){
        if(g_cn_fast_v7_ready[device_id]&&g_cn_fast_v7_enabled[device_id]){launch_cn_fast_v7(stream,threads,count,scratch,ctx);return;}
        if(g_cn_fast_v6_ready[device_id]&&g_cn_fast_v6_enabled[device_id]){launch_cn_fast_v6(stream,threads,count,scratch,ctx);return;}
        if(g_cn_fast_v5_ready[device_id]&&g_cn_fast_v5_enabled[device_id]){launch_cn_fast_v5(stream,threads,count,scratch,ctx);return;}
        if(g_cn_fast_v4_ready[device_id]&&g_cn_fast_v4_enabled[device_id]){launch_cn_fast_v4(stream,threads,count,scratch,ctx);return;}
    }
    int impl=1;if(device_id>=0&&device_id<kMaxLoopTuneDevices&&g_cn_coop_impl_ready[device_id])impl=g_cn_coop_impl[device_id][2];
    launch_cn_coop_impl<2>(stream,impl,threads,count,scratch,ctx);
}

bool validate_cn_fast_v8(int device_id,cudaStream_t stream,std::uint8_t* states,std::uint8_t* scratch,
                         cryptonight::SplitContext* ctx,const CnGeometry& g,int threads,int mode)
{
    constexpr std::size_t n=2U; std::array<std::uint8_t,n*kStateBytes> a{},b{};
    auto run=[&](bool v8,auto& out){
        check_cuda(cudaMemsetAsync(states,0xc1,n*kStateBytes,stream),"CN-Fast v8 parity reset failed");
        const int sb=static_cast<int>((n+g.setup_threads-1U)/g.setup_threads);
        if(g.aes_backend==1)cryptonight_setup_stage<2,true><<<sb,g.setup_threads,0,stream>>>(states,n,scratch,ctx);else cryptonight_setup_stage<2,false><<<sb,g.setup_threads,0,stream>>>(states,n,scratch,ctx);
        check_cuda(cudaGetLastError(),"CN-Fast v8 parity setup failed");
        if(v8)launch_cn_fast_v8(stream,mode,threads,n,scratch,ctx);else launch_cn_fast_existing_for_v8(device_id,stream,threads,n,scratch,ctx);
        check_cuda(cudaGetLastError(),"CN-Fast v8 parity loop failed");
        const int fb=static_cast<int>((n+g.final_threads-1U)/g.final_threads);
        if(g.aes_backend==1)cryptonight_final_stage<2,true><<<fb,g.final_threads,0,stream>>>(states,n,scratch,ctx);else cryptonight_final_stage<2,false><<<fb,g.final_threads,0,stream>>>(states,n,scratch,ctx);
        check_cuda(cudaGetLastError(),"CN-Fast v8 parity final failed");
        check_cuda(cudaMemcpyAsync(out.data(),states,out.size(),cudaMemcpyDeviceToHost,stream),"CN-Fast v8 parity copy failed");
        check_cuda(cudaStreamSynchronize(stream),"CN-Fast v8 parity sync failed");
    };
    run(false,a);run(true,b);return a==b;
}

void ensure_cn_fast_v8_mode(int device_id,const cudaDeviceProp& props,cudaStream_t stream,std::uint8_t* states,
                            std::size_t count,std::uint8_t* scratch,cryptonight::SplitContext* ctx,const CnGeometry& g)
{
    if(device_id<0||device_id>=kMaxLoopTuneDevices||!g_cn_coop_enabled[device_id][2])return;
    if(load_cn_fast_v8_cache(device_id,props,count))return;
    g_cn_fast_v8_mode[device_id]=0;g_cn_fast_v8_ready[device_id]=true;
    int threads=g_cn_coop_threads[device_id][2];if(threads<=0)threads=32;
    // __launch_bounds__(128,...) candidates cannot launch blocks above 128 threads.
    if(threads>128){std::cout<<"[CUDA CN-Fast v8] GPU "<<device_id<<" | threads="<<threads<<" | skipped=launch-bounds>128 | selected=existing\n";save_cn_fast_v8_cache(device_id,props,count);return;}

    auto prep=[&](){check_cuda(cudaMemsetAsync(states,0xd7,count*kStateBytes,stream),"CN-Fast v8 benchmark reset failed");const int sb=static_cast<int>((count+g.setup_threads-1U)/g.setup_threads);if(g.aes_backend==1)cryptonight_setup_stage<2,true><<<sb,g.setup_threads,0,stream>>>(states,count,scratch,ctx);else cryptonight_setup_stage<2,false><<<sb,g.setup_threads,0,stream>>>(states,count,scratch,ctx);check_cuda(cudaGetLastError(),"CN-Fast v8 benchmark setup failed");};

    prep();const float base=time_cuda_phase(stream,[&](){launch_cn_fast_existing_for_v8(device_id,stream,threads,count,scratch,ctx);check_cuda(cudaGetLastError(),"CN-Fast existing-v8 benchmark failed");});

    struct Candidate{int mode;float ms;bool parity;int regs;};
    Candidate cand[2]={{2,0.0F,false,0},{4,0.0F,false,0}};
    for(auto& c:cand){
        c.parity=validate_cn_fast_v8(device_id,stream,states,scratch,ctx,g,threads,c.mode);
        if(c.parity){prep();c.ms=time_cuda_phase(stream,[&](){launch_cn_fast_v8(stream,c.mode,threads,count,scratch,ctx);check_cuda(cudaGetLastError(),"CN-Fast v8 candidate benchmark failed");});}
        cudaFuncAttributes attr{};
        if(c.mode==2){if(cudaFuncGetAttributes(&attr,cryptonight_loop_stage_coop4_v8_b2<2>)==cudaSuccess)c.regs=attr.numRegs;else cudaGetLastError();}
        else {if(cudaFuncGetAttributes(&attr,cryptonight_loop_stage_coop4_v8_b4<2>)==cudaSuccess)c.regs=attr.numRegs;else cudaGetLastError();}
        const float gain=(c.parity&&c.ms>0.0F)?(base/c.ms-1.0F)*100.0F:0.0F;
        std::cout<<std::fixed<<std::setprecision(3)<<"[CUDA CN-Fast v8 candidate] GPU "<<device_id
                 <<" | mode=b"<<c.mode<<" | regs="<<c.regs<<" | baseline="<<base<<" ms | candidate="<<c.ms
                 <<" ms | parity="<<(c.parity?"PASS":"FAIL")<<" | gain="<<gain<<"%"<<std::defaultfloat<<'\n';
    }

    Candidate* best=nullptr;
    for(auto& c:cand)if(c.parity&&c.ms>0.0F&&(!best||c.ms<best->ms))best=&c;
    if(best&&best->ms<base*kCnFastV8RequiredSpeedup)g_cn_fast_v8_mode[device_id]=best->mode;
    const float gain=(best&&best->ms>0.0F)?(base/best->ms-1.0F)*100.0F:0.0F;
    std::cout<<std::fixed<<std::setprecision(3)<<"[CUDA CN-Fast v8] GPU "<<device_id<<" | threads="<<threads
             <<" | baseline="<<base<<" ms | best="<<(best?(best->mode==2?"b2":"b4"):"none")
             <<" | best_ms="<<(best?best->ms:0.0F)<<" | gain="<<gain<<"% | selected="
             <<(g_cn_fast_v8_mode[device_id]==2?"coop4-v8-b2":g_cn_fast_v8_mode[device_id]==4?"coop4-v8-b4":"existing")
             <<std::defaultfloat<<'\n';
    save_cn_fast_v8_cache(device_id,props,count);
}

} // anonymous namespace
