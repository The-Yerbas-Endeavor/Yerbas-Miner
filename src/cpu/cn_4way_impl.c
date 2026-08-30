/* Four-lane CryptoNight execution path.
 * Four independent dependency chains are interleaved. The x86 path shares the
 * direct byte-offset XMM helpers from cn_2way_impl.c so all four chains avoid
 * e2i(), vector spills and repeated address scaling in the hot loop.
 */

static YERBAS_TLS uint8_t* g_yerbas_cn_4way_scratchpad2 = NULL;
static YERBAS_TLS uint8_t* g_yerbas_cn_4way_scratchpad3 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_4way_oaes2 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_4way_oaes3 = NULL;

static int yerbas_cn_4way_resources(void)
{
    if (!yerbas_cn_2way_resources()) return 0;
    if (g_yerbas_cn_4way_scratchpad2 == NULL) g_yerbas_cn_4way_scratchpad2 = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE);
    if (g_yerbas_cn_4way_scratchpad3 == NULL) g_yerbas_cn_4way_scratchpad3 = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE);
    if (g_yerbas_cn_4way_oaes2 == NULL) g_yerbas_cn_4way_oaes2 = oaes_alloc();
    if (g_yerbas_cn_4way_oaes3 == NULL) g_yerbas_cn_4way_oaes3 = oaes_alloc();
    return g_yerbas_cn_4way_scratchpad2 != NULL && g_yerbas_cn_4way_scratchpad3 != NULL &&
           g_yerbas_cn_4way_oaes2 != NULL && g_yerbas_cn_4way_oaes3 != NULL;
}

int yerbas_cn_hash_quad_4way(const char* input0, const char* input1,
                             const char* input2, const char* input3,
                             char* output0, char* output1, char* output2, char* output3,
                             uint32_t len, int variant, uint32_t page_size,
                             uint32_t iterations, size_t aes_rounds)
{
    union cn_slow_hash_state state0, state1, state2, state3;
    uint8_t text0[INIT_SIZE_BYTE], text1[INIT_SIZE_BYTE], text2[INIT_SIZE_BYTE], text3[INIT_SIZE_BYTE];
    uint8_t a0[AES_BLOCK_SIZE], a1[AES_BLOCK_SIZE], a2[AES_BLOCK_SIZE], a3[AES_BLOCK_SIZE];
    uint8_t b0[AES_BLOCK_SIZE * 2], b1[AES_BLOCK_SIZE * 2], b2[AES_BLOCK_SIZE * 2], b3[AES_BLOCK_SIZE * 2];
    uint8_t c0[AES_BLOCK_SIZE], c1[AES_BLOCK_SIZE], c2[AES_BLOCK_SIZE], c3[AES_BLOCK_SIZE];
    uint64_t tweak0, tweak1, tweak2, tweak3;
    uint32_t i;

    if (!input0 || !input1 || !input2 || !input3 || !output0 || !output1 || !output2 || !output3) return 0;
    if (variant != 1 || len < 43U || page_size > YERBAS_CN_MAX_PAGE_SIZE) return 0;
    if (!yerbas_cn_4way_resources()) return 0;

    yerbas_cn_lane_setup(input0,(int)len,&state0,text0,a0,b0,g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_oaes0,page_size);
    yerbas_cn_lane_setup(input1,(int)len,&state1,text1,a1,b1,g_yerbas_cn_2way_scratchpad1,g_yerbas_cn_2way_oaes1,page_size);
    yerbas_cn_lane_setup(input2,(int)len,&state2,text2,a2,b2,g_yerbas_cn_4way_scratchpad2,g_yerbas_cn_4way_oaes2,page_size);
    yerbas_cn_lane_setup(input3,(int)len,&state3,text3,a3,b3,g_yerbas_cn_4way_scratchpad3,g_yerbas_cn_4way_oaes3,page_size);
    tweak0 = *(const uint64_t*)((const uint8_t*)input0 + 35) ^ state0.hs.w[24];
    tweak1 = *(const uint64_t*)((const uint8_t*)input1 + 35) ^ state1.hs.w[24];
    tweak2 = *(const uint64_t*)((const uint8_t*)input2 + 35) ^ state2.hs.w[24];
    tweak3 = *(const uint64_t*)((const uint8_t*)input3 + 35) ^ state3.hs.w[24];

#if YERBAS_X86_AES_RUNTIME
    if (yerbas_runtime_has_aes_avx2()) {
        const uint64_t offset_mask = ((uint64_t)aes_rounds - 1U) << 4;
        __m128i ax0 = _mm_loadu_si128((const __m128i*)a0), ax1 = _mm_loadu_si128((const __m128i*)a1);
        __m128i ax2 = _mm_loadu_si128((const __m128i*)a2), ax3 = _mm_loadu_si128((const __m128i*)a3);
        __m128i bx0 = _mm_loadu_si128((const __m128i*)b0), bx1 = _mm_loadu_si128((const __m128i*)b1);
        __m128i bx2 = _mm_loadu_si128((const __m128i*)b2), bx3 = _mm_loadu_si128((const __m128i*)b3);
        __m128i cx0, cx1, cx2, cx3;
        for (i = 0; i < iterations; ++i) {
            yerbas_cn_lane_loop_phase1_xmm(ax0,bx0,&cx0,g_yerbas_cn_2way_scratchpad0,offset_mask);
            yerbas_cn_lane_loop_phase1_xmm(ax1,bx1,&cx1,g_yerbas_cn_2way_scratchpad1,offset_mask);
            yerbas_cn_lane_loop_phase1_xmm(ax2,bx2,&cx2,g_yerbas_cn_4way_scratchpad2,offset_mask);
            yerbas_cn_lane_loop_phase1_xmm(ax3,bx3,&cx3,g_yerbas_cn_4way_scratchpad3,offset_mask);
            yerbas_cn_lane_loop_phase2_xmm(&ax0,&bx0,cx0,g_yerbas_cn_2way_scratchpad0,offset_mask,tweak0);
            yerbas_cn_lane_loop_phase2_xmm(&ax1,&bx1,cx1,g_yerbas_cn_2way_scratchpad1,offset_mask,tweak1);
            yerbas_cn_lane_loop_phase2_xmm(&ax2,&bx2,cx2,g_yerbas_cn_4way_scratchpad2,offset_mask,tweak2);
            yerbas_cn_lane_loop_phase2_xmm(&ax3,&bx3,cx3,g_yerbas_cn_4way_scratchpad3,offset_mask,tweak3);
        }
    } else
#endif
    {
        for (i = 0; i < iterations; ++i) {
            yerbas_cn_lane_loop_phase1(a0,b0,c0,g_yerbas_cn_2way_scratchpad0,aes_rounds);
            yerbas_cn_lane_loop_phase1(a1,b1,c1,g_yerbas_cn_2way_scratchpad1,aes_rounds);
            yerbas_cn_lane_loop_phase1(a2,b2,c2,g_yerbas_cn_4way_scratchpad2,aes_rounds);
            yerbas_cn_lane_loop_phase1(a3,b3,c3,g_yerbas_cn_4way_scratchpad3,aes_rounds);
            yerbas_cn_lane_loop_phase2(a0,b0,c0,g_yerbas_cn_2way_scratchpad0,aes_rounds,tweak0);
            yerbas_cn_lane_loop_phase2(a1,b1,c1,g_yerbas_cn_2way_scratchpad1,aes_rounds,tweak1);
            yerbas_cn_lane_loop_phase2(a2,b2,c2,g_yerbas_cn_4way_scratchpad2,aes_rounds,tweak2);
            yerbas_cn_lane_loop_phase2(a3,b3,c3,g_yerbas_cn_4way_scratchpad3,aes_rounds,tweak3);
        }
    }

    yerbas_cn_lane_finish(&state0,text0,g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_oaes0,page_size,output0);
    yerbas_cn_lane_finish(&state1,text1,g_yerbas_cn_2way_scratchpad1,g_yerbas_cn_2way_oaes1,page_size,output1);
    yerbas_cn_lane_finish(&state2,text2,g_yerbas_cn_4way_scratchpad2,g_yerbas_cn_4way_oaes2,page_size,output2);
    yerbas_cn_lane_finish(&state3,text3,g_yerbas_cn_4way_scratchpad3,g_yerbas_cn_4way_oaes3,page_size,output3);
    return 1;
}
