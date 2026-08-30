/* Four-lane CryptoNight execution path.
 * Four independent dependency chains are interleaved. CN-Fast has a dedicated
 * fixed-parameter schedule that issues independent scratchpad loads together
 * before AES/multiply work so memory latency can overlap across chains.
 */

static YERBAS_TLS uint8_t* g_yerbas_cn_4way_scratchpad2 = NULL;
static YERBAS_TLS uint8_t* g_yerbas_cn_4way_scratchpad3 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_4way_oaes2 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_4way_oaes3 = NULL;

static int yerbas_cn_4way_resources(void)
{
    if (!yerbas_cn_2way_resources()) return 0;
    if (g_yerbas_cn_4way_scratchpad2 == NULL) g_yerbas_cn_4way_scratchpad2 = yerbas_cn_alloc_scratchpad();
    if (g_yerbas_cn_4way_scratchpad3 == NULL) g_yerbas_cn_4way_scratchpad3 = yerbas_cn_alloc_scratchpad();
    if (g_yerbas_cn_4way_oaes2 == NULL) g_yerbas_cn_4way_oaes2 = oaes_alloc();
    if (g_yerbas_cn_4way_oaes3 == NULL) g_yerbas_cn_4way_oaes3 = oaes_alloc();
    return g_yerbas_cn_4way_scratchpad2 != NULL && g_yerbas_cn_4way_scratchpad3 != NULL &&
           g_yerbas_cn_4way_oaes2 != NULL && g_yerbas_cn_4way_oaes3 != NULL;
}

#if YERBAS_X86_AES_RUNTIME
YERBAS_AES_TARGET
static void yerbas_cn_fast_quad_loop(__m128i* ax0p, __m128i* bx0p,
                                     __m128i* ax1p, __m128i* bx1p,
                                     __m128i* ax2p, __m128i* bx2p,
                                     __m128i* ax3p, __m128i* bx3p,
                                     uint8_t* sp0, uint8_t* sp1,
                                     uint8_t* sp2, uint8_t* sp3,
                                     uint64_t tweak0, uint64_t tweak1,
                                     uint64_t tweak2, uint64_t tweak3)
{
    __m128i ax0=*ax0p,bx0=*bx0p,ax1=*ax1p,bx1=*bx1p;
    __m128i ax2=*ax2p,bx2=*bx2p,ax3=*ax3p,bx3=*bx3p;
    uint32_t i;
    for (i = 0; i < YERBAS_CN_FAST_ITERATIONS; ++i) {
        __m128i* p00=(__m128i*)(sp0+(yerbas_xmm_lo64(ax0)&YERBAS_CN_FAST_OFFSET_MASK));
        __m128i* p01=(__m128i*)(sp1+(yerbas_xmm_lo64(ax1)&YERBAS_CN_FAST_OFFSET_MASK));
        __m128i* p02=(__m128i*)(sp2+(yerbas_xmm_lo64(ax2)&YERBAS_CN_FAST_OFFSET_MASK));
        __m128i* p03=(__m128i*)(sp3+(yerbas_xmm_lo64(ax3)&YERBAS_CN_FAST_OFFSET_MASK));
        const __m128i s0=_mm_load_si128(p00),s1=_mm_load_si128(p01),s2=_mm_load_si128(p02),s3=_mm_load_si128(p03);
        const __m128i cx0=_mm_aesenc_si128(s0,ax0),cx1=_mm_aesenc_si128(s1,ax1);
        const __m128i cx2=_mm_aesenc_si128(s2,ax2),cx3=_mm_aesenc_si128(s3,ax3);
        _mm_store_si128(p00,yerbas_cn_variant1_store_tweak_xmm(_mm_xor_si128(cx0,bx0)));
        _mm_store_si128(p01,yerbas_cn_variant1_store_tweak_xmm(_mm_xor_si128(cx1,bx1)));
        _mm_store_si128(p02,yerbas_cn_variant1_store_tweak_xmm(_mm_xor_si128(cx2,bx2)));
        _mm_store_si128(p03,yerbas_cn_variant1_store_tweak_xmm(_mm_xor_si128(cx3,bx3)));
        {
            uint64_t* d0=(uint64_t*)(sp0+(yerbas_xmm_lo64(cx0)&YERBAS_CN_FAST_OFFSET_MASK));
            uint64_t* d1=(uint64_t*)(sp1+(yerbas_xmm_lo64(cx1)&YERBAS_CN_FAST_OFFSET_MASK));
            uint64_t* d2=(uint64_t*)(sp2+(yerbas_xmm_lo64(cx2)&YERBAS_CN_FAST_OFFSET_MASK));
            uint64_t* d3=(uint64_t*)(sp3+(yerbas_xmm_lo64(cx3)&YERBAS_CN_FAST_OFFSET_MASK));
            const uint64_t t00=d0[0],t01=d0[1],t10=d1[0],t11=d1[1];
            const uint64_t t20=d2[0],t21=d2[1],t30=d3[0],t31=d3[1];
            uint64_t hi0,hi1,hi2,hi3;
            const uint64_t lo0=mul128(yerbas_xmm_lo64(cx0),t00,&hi0);
            const uint64_t lo1=mul128(yerbas_xmm_lo64(cx1),t10,&hi1);
            const uint64_t lo2=mul128(yerbas_xmm_lo64(cx2),t20,&hi2);
            const uint64_t lo3=mul128(yerbas_xmm_lo64(cx3),t30,&hi3);
            uint64_t a00=yerbas_xmm_lo64(ax0)+hi0,a01=yerbas_xmm_hi64(ax0)+lo0;
            uint64_t a10=yerbas_xmm_lo64(ax1)+hi1,a11=yerbas_xmm_hi64(ax1)+lo1;
            uint64_t a20=yerbas_xmm_lo64(ax2)+hi2,a21=yerbas_xmm_hi64(ax2)+lo2;
            uint64_t a30=yerbas_xmm_lo64(ax3)+hi3,a31=yerbas_xmm_hi64(ax3)+lo3;
            d0[0]=a00;d0[1]=a01^tweak0;d1[0]=a10;d1[1]=a11^tweak1;
            d2[0]=a20;d2[1]=a21^tweak2;d3[0]=a30;d3[1]=a31^tweak3;
            a00^=t00;a01^=t01;a10^=t10;a11^=t11;a20^=t20;a21^=t21;a30^=t30;a31^=t31;
            ax0=_mm_set_epi64x((long long)a01,(long long)a00);ax1=_mm_set_epi64x((long long)a11,(long long)a10);
            ax2=_mm_set_epi64x((long long)a21,(long long)a20);ax3=_mm_set_epi64x((long long)a31,(long long)a30);
            bx0=cx0;bx1=cx1;bx2=cx2;bx3=cx3;
        }
    }
    *ax0p=ax0;*bx0p=bx0;*ax1p=ax1;*bx1p=bx1;*ax2p=ax2;*bx2p=bx2;*ax3p=ax3;*bx3p=bx3;
}
#endif

int yerbas_cn_hash_quad_4way(const char* input0,const char* input1,const char* input2,const char* input3,
                             char* output0,char* output1,char* output2,char* output3,
                             uint32_t len,int variant,uint32_t page_size,uint32_t iterations,size_t aes_rounds)
{
    union cn_slow_hash_state state0,state1,state2,state3;
    uint8_t text0[INIT_SIZE_BYTE],text1[INIT_SIZE_BYTE],text2[INIT_SIZE_BYTE],text3[INIT_SIZE_BYTE];
    uint8_t a0[AES_BLOCK_SIZE],a1[AES_BLOCK_SIZE],a2[AES_BLOCK_SIZE],a3[AES_BLOCK_SIZE];
    uint8_t b0[AES_BLOCK_SIZE*2],b1[AES_BLOCK_SIZE*2],b2[AES_BLOCK_SIZE*2],b3[AES_BLOCK_SIZE*2];
    uint8_t c0[AES_BLOCK_SIZE],c1[AES_BLOCK_SIZE],c2[AES_BLOCK_SIZE],c3[AES_BLOCK_SIZE];
    uint64_t tweak0,tweak1,tweak2,tweak3; uint32_t i;
    if(!input0||!input1||!input2||!input3||!output0||!output1||!output2||!output3)return 0;
    if(variant!=1||len<43U||page_size>YERBAS_CN_MAX_PAGE_SIZE)return 0;
    if(!yerbas_cn_4way_resources())return 0;
    yerbas_cn_lane_setup(input0,(int)len,&state0,text0,a0,b0,g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_oaes0,page_size);
    yerbas_cn_lane_setup(input1,(int)len,&state1,text1,a1,b1,g_yerbas_cn_2way_scratchpad1,g_yerbas_cn_2way_oaes1,page_size);
    yerbas_cn_lane_setup(input2,(int)len,&state2,text2,a2,b2,g_yerbas_cn_4way_scratchpad2,g_yerbas_cn_4way_oaes2,page_size);
    yerbas_cn_lane_setup(input3,(int)len,&state3,text3,a3,b3,g_yerbas_cn_4way_scratchpad3,g_yerbas_cn_4way_oaes3,page_size);
    tweak0=*(const uint64_t*)((const uint8_t*)input0+35)^state0.hs.w[24];tweak1=*(const uint64_t*)((const uint8_t*)input1+35)^state1.hs.w[24];
    tweak2=*(const uint64_t*)((const uint8_t*)input2+35)^state2.hs.w[24];tweak3=*(const uint64_t*)((const uint8_t*)input3+35)^state3.hs.w[24];
#if YERBAS_X86_AES_RUNTIME
    if(yerbas_runtime_has_aes_avx2()){
        const uint64_t offset_mask=((uint64_t)aes_rounds-1U)<<4;
        __m128i ax0=_mm_loadu_si128((const __m128i*)a0),ax1=_mm_loadu_si128((const __m128i*)a1),ax2=_mm_loadu_si128((const __m128i*)a2),ax3=_mm_loadu_si128((const __m128i*)a3);
        __m128i bx0=_mm_loadu_si128((const __m128i*)b0),bx1=_mm_loadu_si128((const __m128i*)b1),bx2=_mm_loadu_si128((const __m128i*)b2),bx3=_mm_loadu_si128((const __m128i*)b3);
        if(page_size==2097152U&&iterations==YERBAS_CN_FAST_ITERATIONS&&aes_rounds==131072U)
            yerbas_cn_fast_quad_loop(&ax0,&bx0,&ax1,&bx1,&ax2,&bx2,&ax3,&bx3,g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_scratchpad1,g_yerbas_cn_4way_scratchpad2,g_yerbas_cn_4way_scratchpad3,tweak0,tweak1,tweak2,tweak3);
        else for(i=0;i<iterations;++i){__m128i cx0,cx1,cx2,cx3;
            yerbas_cn_lane_loop_phase1_xmm(ax0,bx0,&cx0,g_yerbas_cn_2way_scratchpad0,offset_mask);yerbas_cn_lane_loop_phase1_xmm(ax1,bx1,&cx1,g_yerbas_cn_2way_scratchpad1,offset_mask);
            yerbas_cn_lane_loop_phase1_xmm(ax2,bx2,&cx2,g_yerbas_cn_4way_scratchpad2,offset_mask);yerbas_cn_lane_loop_phase1_xmm(ax3,bx3,&cx3,g_yerbas_cn_4way_scratchpad3,offset_mask);
            yerbas_cn_lane_loop_phase2_xmm(&ax0,&bx0,cx0,g_yerbas_cn_2way_scratchpad0,offset_mask,tweak0);yerbas_cn_lane_loop_phase2_xmm(&ax1,&bx1,cx1,g_yerbas_cn_2way_scratchpad1,offset_mask,tweak1);
            yerbas_cn_lane_loop_phase2_xmm(&ax2,&bx2,cx2,g_yerbas_cn_4way_scratchpad2,offset_mask,tweak2);yerbas_cn_lane_loop_phase2_xmm(&ax3,&bx3,cx3,g_yerbas_cn_4way_scratchpad3,offset_mask,tweak3);}
        _mm_storeu_si128((__m128i*)a0,ax0);_mm_storeu_si128((__m128i*)a1,ax1);_mm_storeu_si128((__m128i*)a2,ax2);_mm_storeu_si128((__m128i*)a3,ax3);
        _mm_storeu_si128((__m128i*)b0,bx0);_mm_storeu_si128((__m128i*)b1,bx1);_mm_storeu_si128((__m128i*)b2,bx2);_mm_storeu_si128((__m128i*)b3,bx3);
    }else
#endif
    for(i=0;i<iterations;++i){yerbas_cn_lane_loop_phase1(a0,b0,c0,g_yerbas_cn_2way_scratchpad0,aes_rounds);yerbas_cn_lane_loop_phase1(a1,b1,c1,g_yerbas_cn_2way_scratchpad1,aes_rounds);yerbas_cn_lane_loop_phase1(a2,b2,c2,g_yerbas_cn_4way_scratchpad2,aes_rounds);yerbas_cn_lane_loop_phase1(a3,b3,c3,g_yerbas_cn_4way_scratchpad3,aes_rounds);yerbas_cn_lane_loop_phase2(a0,b0,c0,g_yerbas_cn_2way_scratchpad0,aes_rounds,tweak0);yerbas_cn_lane_loop_phase2(a1,b1,c1,g_yerbas_cn_2way_scratchpad1,aes_rounds,tweak1);yerbas_cn_lane_loop_phase2(a2,b2,c2,g_yerbas_cn_4way_scratchpad2,aes_rounds,tweak2);yerbas_cn_lane_loop_phase2(a3,b3,c3,g_yerbas_cn_4way_scratchpad3,aes_rounds,tweak3);}
    yerbas_cn_lane_finish(&state0,text0,g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_oaes0,page_size,output0);yerbas_cn_lane_finish(&state1,text1,g_yerbas_cn_2way_scratchpad1,g_yerbas_cn_2way_oaes1,page_size,output1);yerbas_cn_lane_finish(&state2,text2,g_yerbas_cn_4way_scratchpad2,g_yerbas_cn_4way_oaes2,page_size,output2);yerbas_cn_lane_finish(&state3,text3,g_yerbas_cn_4way_scratchpad3,g_yerbas_cn_4way_oaes3,page_size,output3);return 1;
}
