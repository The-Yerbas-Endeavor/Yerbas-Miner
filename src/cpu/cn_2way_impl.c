/* Genuine two-lane CryptoNight execution path.
 *
 * The production x86 path keeps A/B/C in XMM registers and uses byte offsets
 * directly. GhostRider CN address spaces are powers of two, so
 * e2i(x,aes_rounds)*16 is exactly low64(x) & ((aes_rounds-1)<<4). CN-Fast has
 * an additional fixed-parameter loop that issues independent chain loads
 * together so cache/DRAM latency can overlap across lanes.
 */

static YERBAS_TLS uint8_t* g_yerbas_cn_2way_scratchpad0 = NULL;
static YERBAS_TLS uint8_t* g_yerbas_cn_2way_scratchpad1 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_2way_oaes0 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_2way_oaes1 = NULL;

static int yerbas_cn_2way_resources(void)
{
    if (g_yerbas_cn_2way_scratchpad0 == NULL)
        g_yerbas_cn_2way_scratchpad0 = yerbas_cn_alloc_scratchpad();
    if (g_yerbas_cn_2way_scratchpad1 == NULL)
        g_yerbas_cn_2way_scratchpad1 = yerbas_cn_alloc_scratchpad();
    if (g_yerbas_cn_2way_oaes0 == NULL) g_yerbas_cn_2way_oaes0 = oaes_alloc();
    if (g_yerbas_cn_2way_oaes1 == NULL) g_yerbas_cn_2way_oaes1 = oaes_alloc();
    return g_yerbas_cn_2way_scratchpad0 != NULL &&
           g_yerbas_cn_2way_scratchpad1 != NULL &&
           g_yerbas_cn_2way_oaes0 != NULL &&
           g_yerbas_cn_2way_oaes1 != NULL;
}

static void yerbas_cn_lane_setup(const char* input,
                                 int len,
                                 union cn_slow_hash_state* state,
                                 uint8_t text[INIT_SIZE_BYTE],
                                 uint8_t a[AES_BLOCK_SIZE],
                                 uint8_t b[AES_BLOCK_SIZE * 2],
                                 uint8_t* long_state,
                                 OAES_CTX* aes_ctx,
                                 uint32_t page_size)
{
    uint8_t aes_key[AES_KEY_SIZE];
    const size_t init_rounds = page_size / INIT_SIZE_BYTE;
    size_t i, j;
    hash_process(&state->hs, (const uint8_t*)input, len);
    memcpy(text, state->init, INIT_SIZE_BYTE);
    memcpy(aes_key, state->hs.b, AES_KEY_SIZE);
    oaes_key_import_data((oaes_ctx*)aes_ctx, aes_key, AES_KEY_SIZE);
    for (i = 0; i < init_rounds; ++i) {
#if YERBAS_X86_AES_RUNTIME
        if (yerbas_runtime_has_aes_avx2()) {
            yerbas_aesni_pseudo_round8(text, ((oaes_ctx*)aes_ctx)->key->exp_data);
        } else
#endif
        {
            for (j = 0; j < INIT_SIZE_BLK; ++j)
                yerbas_selected_pseudo_round(&text[AES_BLOCK_SIZE * j],
                                             &text[AES_BLOCK_SIZE * j],
                                             ((oaes_ctx*)aes_ctx)->key->exp_data);
        }
        memcpy(&long_state[i * INIT_SIZE_BYTE], text, INIT_SIZE_BYTE);
    }
    for (i = 0; i < 16; ++i) {
        a[i] = state->k[i] ^ state->k[32 + i];
        b[i] = state->k[16 + i] ^ state->k[48 + i];
    }
    memset(b + AES_BLOCK_SIZE, 0, AES_BLOCK_SIZE);
}

static inline void yerbas_cn_lane_loop_phase1(uint8_t a[AES_BLOCK_SIZE],
                                               uint8_t b[AES_BLOCK_SIZE * 2],
                                               uint8_t c[AES_BLOCK_SIZE],
                                               uint8_t* long_state,
                                               size_t aes_rounds)
{
    const size_t j = e2i(a, aes_rounds);
    yerbas_selected_single_round(&long_state[j * AES_BLOCK_SIZE], c, a);
    xor_blocks_dst(c, b, &long_state[j * AES_BLOCK_SIZE]);
    {
        const uint8_t tmp = long_state[j * AES_BLOCK_SIZE + 11];
        static const uint32_t table = 0x75310;
        const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1;
        long_state[j * AES_BLOCK_SIZE + 11] = tmp ^ ((table >> index) & 0x30);
    }
}

static inline void yerbas_cn_lane_loop_phase2(uint8_t a[AES_BLOCK_SIZE],
                                               uint8_t b[AES_BLOCK_SIZE * 2],
                                               const uint8_t c[AES_BLOCK_SIZE],
                                               uint8_t* long_state,
                                               size_t aes_rounds,
                                               uint64_t tweak1_2)
{
    const size_t j = e2i(c, aes_rounds);
    uint64_t* dst = (uint64_t*)&long_state[j * AES_BLOCK_SIZE];
    const uint64_t t0 = dst[0], t1 = dst[1];
    uint64_t hi;
    const uint64_t lo = mul128(((const uint64_t*)c)[0], t0, &hi);
    ((uint64_t*)a)[0] += hi;
    ((uint64_t*)a)[1] += lo;
    dst[0] = ((uint64_t*)a)[0];
    dst[1] = ((uint64_t*)a)[1] ^ tweak1_2;
    ((uint64_t*)a)[0] ^= t0;
    ((uint64_t*)a)[1] ^= t1;
    copy_block(b + AES_BLOCK_SIZE, b);
    copy_block(b, c);
}

#if YERBAS_X86_AES_RUNTIME
YERBAS_AES_TARGET
static inline uint64_t yerbas_xmm_lo64(__m128i x)
{
    return (uint64_t)_mm_cvtsi128_si64(x);
}

YERBAS_AES_TARGET
static inline uint64_t yerbas_xmm_hi64(__m128i x)
{
    return (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(x, 8));
}

YERBAS_AES_TARGET
static inline __m128i yerbas_cn_variant1_store_tweak_xmm(__m128i value)
{
    static const uint32_t table = 0x75310;
    const uint64_t high = yerbas_xmm_hi64(value);
    const uint8_t tmp = (uint8_t)(high >> 24);
    const uint8_t index = (uint8_t)((((tmp >> 3) & 6U) | (tmp & 1U)) << 1);
    const uint64_t delta = (uint64_t)((table >> index) & 0x30U) << 24;
    return _mm_xor_si128(value, _mm_set_epi64x((long long)delta, 0));
}

YERBAS_AES_TARGET
static inline void yerbas_cn_lane_loop_phase1_xmm(__m128i a,
                                                   __m128i b,
                                                   __m128i* c,
                                                   uint8_t* long_state,
                                                   uint64_t offset_mask)
{
    const uint64_t off = yerbas_xmm_lo64(a) & offset_mask;
    __m128i* p = (__m128i*)(long_state + off);
    const __m128i x = _mm_aesenc_si128(_mm_load_si128(p), a);
    *c = x;
    _mm_store_si128(p, yerbas_cn_variant1_store_tweak_xmm(_mm_xor_si128(x, b)));
    _mm_prefetch((const char*)(long_state + (yerbas_xmm_lo64(x) & offset_mask)), _MM_HINT_T0);
}

YERBAS_AES_TARGET
static inline void yerbas_cn_lane_loop_phase2_xmm(__m128i* a,
                                                   __m128i* b,
                                                   __m128i c,
                                                   uint8_t* long_state,
                                                   uint64_t offset_mask,
                                                   uint64_t tweak1_2)
{
    const uint64_t off = yerbas_xmm_lo64(c) & offset_mask;
    uint64_t* dst = (uint64_t*)(long_state + off);
    const uint64_t t0 = dst[0], t1 = dst[1];
    uint64_t hi;
    const uint64_t lo = mul128(yerbas_xmm_lo64(c), t0, &hi);
    uint64_t a0 = yerbas_xmm_lo64(*a) + hi;
    uint64_t a1 = yerbas_xmm_hi64(*a) + lo;
    dst[0] = a0;
    dst[1] = a1 ^ tweak1_2;
    a0 ^= t0;
    a1 ^= t1;
    *a = _mm_set_epi64x((long long)a1, (long long)a0);
    *b = c;
    _mm_prefetch((const char*)(long_state + (a0 & offset_mask)), _MM_HINT_T0);
}

#define YERBAS_CN_FAST_OFFSET_MASK 2097136ULL
#define YERBAS_CN_FAST_ITERATIONS 262144U

YERBAS_AES_TARGET
static void yerbas_cn_fast_pair_loop(__m128i* ax0p, __m128i* bx0p,
                                     __m128i* ax1p, __m128i* bx1p,
                                     uint8_t* sp0, uint8_t* sp1,
                                     uint64_t tweak0, uint64_t tweak1)
{
    __m128i ax0 = *ax0p, bx0 = *bx0p, ax1 = *ax1p, bx1 = *bx1p;
    uint32_t i;
    for (i = 0; i < YERBAS_CN_FAST_ITERATIONS; ++i) {
        __m128i* p00 = (__m128i*)(sp0 + (yerbas_xmm_lo64(ax0) & YERBAS_CN_FAST_OFFSET_MASK));
        __m128i* p01 = (__m128i*)(sp1 + (yerbas_xmm_lo64(ax1) & YERBAS_CN_FAST_OFFSET_MASK));
        const __m128i s0 = _mm_load_si128(p00);
        const __m128i s1 = _mm_load_si128(p01);
        const __m128i cx0 = _mm_aesenc_si128(s0, ax0);
        const __m128i cx1 = _mm_aesenc_si128(s1, ax1);
        _mm_store_si128(p00, yerbas_cn_variant1_store_tweak_xmm(_mm_xor_si128(cx0, bx0)));
        _mm_store_si128(p01, yerbas_cn_variant1_store_tweak_xmm(_mm_xor_si128(cx1, bx1)));

        {
            uint64_t* d0 = (uint64_t*)(sp0 + (yerbas_xmm_lo64(cx0) & YERBAS_CN_FAST_OFFSET_MASK));
            uint64_t* d1 = (uint64_t*)(sp1 + (yerbas_xmm_lo64(cx1) & YERBAS_CN_FAST_OFFSET_MASK));
            const uint64_t t00 = d0[0], t01 = d0[1];
            const uint64_t t10 = d1[0], t11 = d1[1];
            uint64_t hi0, hi1;
            const uint64_t lo0 = mul128(yerbas_xmm_lo64(cx0), t00, &hi0);
            const uint64_t lo1 = mul128(yerbas_xmm_lo64(cx1), t10, &hi1);
            uint64_t a00 = yerbas_xmm_lo64(ax0) + hi0;
            uint64_t a01 = yerbas_xmm_hi64(ax0) + lo0;
            uint64_t a10 = yerbas_xmm_lo64(ax1) + hi1;
            uint64_t a11 = yerbas_xmm_hi64(ax1) + lo1;
            d0[0] = a00; d0[1] = a01 ^ tweak0;
            d1[0] = a10; d1[1] = a11 ^ tweak1;
            a00 ^= t00; a01 ^= t01;
            a10 ^= t10; a11 ^= t11;
            ax0 = _mm_set_epi64x((long long)a01, (long long)a00);
            ax1 = _mm_set_epi64x((long long)a11, (long long)a10);
            bx0 = cx0; bx1 = cx1;
        }
    }
    *ax0p = ax0; *bx0p = bx0; *ax1p = ax1; *bx1p = bx1;
}
#endif

static void yerbas_cn_lane_finish(union cn_slow_hash_state* state,
                                  uint8_t text[INIT_SIZE_BYTE],
                                  uint8_t* long_state,
                                  OAES_CTX* aes_ctx,
                                  uint32_t page_size,
                                  char* output)
{
    const size_t init_rounds = page_size / INIT_SIZE_BYTE;
    size_t i, j;
    memcpy(text, state->init, INIT_SIZE_BYTE);
    oaes_key_import_data((oaes_ctx*)aes_ctx, &state->hs.b[32], AES_KEY_SIZE);
    for (i = 0; i < init_rounds; ++i) {
#if YERBAS_X86_AES_RUNTIME
        if (yerbas_runtime_has_aes_avx2()) {
            yerbas_aesni_xor_pseudo_round8(text, &long_state[i * INIT_SIZE_BYTE],
                                           ((oaes_ctx*)aes_ctx)->key->exp_data);
        } else
#endif
        {
            for (j = 0; j < INIT_SIZE_BLK; ++j) {
                xor_blocks(&text[j * AES_BLOCK_SIZE],
                           &long_state[i * INIT_SIZE_BYTE + j * AES_BLOCK_SIZE]);
                yerbas_selected_pseudo_round(&text[j * AES_BLOCK_SIZE],
                                             &text[j * AES_BLOCK_SIZE],
                                             ((oaes_ctx*)aes_ctx)->key->exp_data);
            }
        }
    }
    memcpy(state->init, text, INIT_SIZE_BYTE);
    hash_permutation(&state->hs);
    extra_hashes[state->hs.b[0] & 3](state, 200, output);
}

int yerbas_cn_hash_pair_2way(const char* input0,
                             const char* input1,
                             char* output0,
                             char* output1,
                             uint32_t len,
                             int variant,
                             uint32_t page_size,
                             uint32_t iterations,
                             size_t aes_rounds)
{
    union cn_slow_hash_state state0, state1;
    uint8_t text0[INIT_SIZE_BYTE], text1[INIT_SIZE_BYTE];
    uint8_t a0[AES_BLOCK_SIZE], a1[AES_BLOCK_SIZE];
    uint8_t b0[AES_BLOCK_SIZE * 2], b1[AES_BLOCK_SIZE * 2];
    uint8_t c0[AES_BLOCK_SIZE], c1[AES_BLOCK_SIZE];
    uint64_t tweak0, tweak1;
    uint32_t i;
    if (!input0 || !input1 || !output0 || !output1) return 0;
    if (variant != 1 || len < 43U || page_size > YERBAS_CN_MAX_PAGE_SIZE) return 0;
    if (!yerbas_cn_2way_resources()) return 0;
    yerbas_cn_lane_setup(input0,(int)len,&state0,text0,a0,b0,g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_oaes0,page_size);
    yerbas_cn_lane_setup(input1,(int)len,&state1,text1,a1,b1,g_yerbas_cn_2way_scratchpad1,g_yerbas_cn_2way_oaes1,page_size);
    tweak0 = *(const uint64_t*)((const uint8_t*)input0 + 35) ^ state0.hs.w[24];
    tweak1 = *(const uint64_t*)((const uint8_t*)input1 + 35) ^ state1.hs.w[24];
#if YERBAS_X86_AES_RUNTIME
    if (yerbas_runtime_has_aes_avx2()) {
        const uint64_t offset_mask = ((uint64_t)aes_rounds - 1U) << 4;
        __m128i ax0 = _mm_loadu_si128((const __m128i*)a0), ax1 = _mm_loadu_si128((const __m128i*)a1);
        __m128i bx0 = _mm_loadu_si128((const __m128i*)b0), bx1 = _mm_loadu_si128((const __m128i*)b1);
        if (page_size == 2097152U && iterations == YERBAS_CN_FAST_ITERATIONS && aes_rounds == 131072U) {
            yerbas_cn_fast_pair_loop(&ax0,&bx0,&ax1,&bx1,
                                     g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_scratchpad1,
                                     tweak0,tweak1);
        } else {
            for (i=0;i<iterations;++i) {
                __m128i cx0,cx1;
                yerbas_cn_lane_loop_phase1_xmm(ax0,bx0,&cx0,g_yerbas_cn_2way_scratchpad0,offset_mask);
                yerbas_cn_lane_loop_phase1_xmm(ax1,bx1,&cx1,g_yerbas_cn_2way_scratchpad1,offset_mask);
                yerbas_cn_lane_loop_phase2_xmm(&ax0,&bx0,cx0,g_yerbas_cn_2way_scratchpad0,offset_mask,tweak0);
                yerbas_cn_lane_loop_phase2_xmm(&ax1,&bx1,cx1,g_yerbas_cn_2way_scratchpad1,offset_mask,tweak1);
            }
        }
        _mm_storeu_si128((__m128i*)a0,ax0); _mm_storeu_si128((__m128i*)a1,ax1);
        _mm_storeu_si128((__m128i*)b0,bx0); _mm_storeu_si128((__m128i*)b1,bx1);
    } else
#endif
    {
        for (i=0;i<iterations;++i) {
            yerbas_cn_lane_loop_phase1(a0,b0,c0,g_yerbas_cn_2way_scratchpad0,aes_rounds);
            yerbas_cn_lane_loop_phase1(a1,b1,c1,g_yerbas_cn_2way_scratchpad1,aes_rounds);
            yerbas_cn_lane_loop_phase2(a0,b0,c0,g_yerbas_cn_2way_scratchpad0,aes_rounds,tweak0);
            yerbas_cn_lane_loop_phase2(a1,b1,c1,g_yerbas_cn_2way_scratchpad1,aes_rounds,tweak1);
        }
    }
    yerbas_cn_lane_finish(&state0,text0,g_yerbas_cn_2way_scratchpad0,g_yerbas_cn_2way_oaes0,page_size,output0);
    yerbas_cn_lane_finish(&state1,text1,g_yerbas_cn_2way_scratchpad1,g_yerbas_cn_2way_oaes1,page_size,output1);
    return 1;
}

#include "cn_4way_impl.c"
