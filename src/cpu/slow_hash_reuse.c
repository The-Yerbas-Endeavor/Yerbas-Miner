#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "slow-hash.h"
#include "oaes_lib.h"

void aesb_single_round(const uint8_t* in, uint8_t* out, uint8_t* expanded_key);
void aesb_pseudo_round(const uint8_t* in, uint8_t* out, uint8_t* expanded_key);

#if defined(_MSC_VER)
#include <intrin.h>
#define YERBAS_TLS __declspec(thread)
#else
#define YERBAS_TLS _Thread_local
#endif

#define YERBAS_CN_MAX_PAGE_SIZE 2097152u

static YERBAS_TLS uint8_t* g_yerbas_cn_scratchpad = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_oaes = NULL;
static YERBAS_TLS int g_yerbas_aes_avx2_runtime = -1;

static void* yerbas_tls_cn_malloc(size_t requested)
{
    if (requested > YERBAS_CN_MAX_PAGE_SIZE) return NULL;
    if (g_yerbas_cn_scratchpad == NULL)
        g_yerbas_cn_scratchpad = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE);
    return g_yerbas_cn_scratchpad;
}

static void yerbas_tls_cn_free(void* ptr) { (void)ptr; }

static OAES_CTX* yerbas_tls_oaes_alloc(void)
{
    if (g_yerbas_cn_oaes == NULL) g_yerbas_cn_oaes = oaes_alloc();
    return g_yerbas_cn_oaes;
}

static void yerbas_tls_oaes_free(OAES_CTX** ctx)
{
    if (ctx != NULL) *ctx = NULL;
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <wmmintrin.h>
#define YERBAS_X86_AES_RUNTIME 1
#else
#define YERBAS_X86_AES_RUNTIME 0
#endif

#if YERBAS_X86_AES_RUNTIME && (defined(__GNUC__) || defined(__clang__))
#define YERBAS_AES_TARGET __attribute__((target("aes,avx2,bmi2,sse4.2")))
#else
#define YERBAS_AES_TARGET
#endif

static int yerbas_detect_aes_avx2(void)
{
#if YERBAS_X86_AES_RUNTIME && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("aes") && __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("sse4.2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    if ((regs[2] & (1 << 25)) == 0 || (regs[2] & (1 << 20)) == 0 ||
        (regs[2] & (1 << 27)) == 0 || (regs[2] & (1 << 28)) == 0) return 0;
    if ((_xgetbv(0) & 0x6) != 0x6) return 0;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0 && (regs[1] & (1 << 8)) != 0;
#else
    return 0;
#endif
}

static int yerbas_runtime_has_aes_avx2(void)
{
    if (g_yerbas_aes_avx2_runtime < 0)
        g_yerbas_aes_avx2_runtime = yerbas_detect_aes_avx2() ? 1 : 0;
    return g_yerbas_aes_avx2_runtime;
}

#if YERBAS_X86_AES_RUNTIME
YERBAS_AES_TARGET
static int yerbas_aesni_single_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
    const __m128i state = _mm_loadu_si128((const __m128i*)in);
    const __m128i key = _mm_loadu_si128((const __m128i*)expanded_key);
    _mm_storeu_si128((__m128i*)out, _mm_aesenc_si128(state, key));
    return 0;
}

YERBAS_AES_TARGET
static int yerbas_aesni_pseudo_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
    __m128i state = _mm_loadu_si128((const __m128i*)in);
    int round;
    for (round = 0; round < 10; ++round) {
        const __m128i key = _mm_loadu_si128((const __m128i*)(expanded_key + round * 16));
        state = _mm_aesenc_si128(state, key);
    }
    _mm_storeu_si128((__m128i*)out, state);
    return 0;
}
#endif

static int yerbas_selected_single_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
#if YERBAS_X86_AES_RUNTIME
    if (yerbas_runtime_has_aes_avx2()) return yerbas_aesni_single_round(in, out, expanded_key);
#endif
    aesb_single_round(in, out, (uint8_t*)expanded_key);
    return 0;
}

static int yerbas_selected_pseudo_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
#if YERBAS_X86_AES_RUNTIME
    if (yerbas_runtime_has_aes_avx2()) return yerbas_aesni_pseudo_round(in, out, expanded_key);
#endif
    aesb_pseudo_round(in, out, (uint8_t*)expanded_key);
    return 0;
}

const char* yerbas_cn_reuse_backend(void)
{
    return yerbas_runtime_has_aes_avx2() ? "aes-ni+avx2-runtime" : "portable-table-aes";
}

unsigned int yerbas_cn_reuse_compile_features(void)
{
    unsigned int features = 0;
    if (yerbas_runtime_has_aes_avx2()) features = 0x1fu;
    return features;
}

#define cn_slow_hash yerbas_cn_slow_hash_reuse_impl
#define cn_fast_hash yerbas_cn_fast_hash_reuse_impl
#define do_groestl_hash yerbas_reuse_do_groestl_hash
#define malloc yerbas_tls_cn_malloc
#define free yerbas_tls_cn_free
#if defined(_MSC_VER)
#define _malloca yerbas_tls_cn_malloc
#define _freea yerbas_tls_cn_free
#endif
#define oaes_alloc yerbas_tls_oaes_alloc
#define oaes_free yerbas_tls_oaes_free
#define aesb_single_round yerbas_selected_single_round
#define aesb_pseudo_round yerbas_selected_pseudo_round

#include "slow-hash.c"

#undef cn_slow_hash
#undef cn_fast_hash
#undef do_groestl_hash
#undef malloc
#undef free
#if defined(_MSC_VER)
#undef _malloca
#undef _freea
#endif
#undef oaes_alloc
#undef oaes_free
#undef aesb_single_round
#undef aesb_pseudo_round

#if YERBAS_X86_AES_RUNTIME
#include "cn_aesni_parallel.inc"

/* The six Yerbas GhostRider CryptoNight flavours are fixed parameter sets.
 * Generate one scalar variant-1 AES-NI kernel per set so the dependency-heavy
 * loop contains no e2i() call, no runtime page/iteration arithmetic and no
 * generic AES/variant dispatch. A/B/C stay in XMM registers; only the two
 * data-dependent scratchpad locations touch memory each iteration. */
#define YERBAS_DEFINE_CN_V1_KERNEL(NAME, PAGE_SIZE_CONST, ITERATIONS_CONST, OFFSET_MASK_CONST) \
YERBAS_AES_TARGET \
static void NAME(const char* input, char* output, uint32_t len) \
{ \
    union cn_slow_hash_state state; \
    uint8_t text[INIT_SIZE_BYTE]; \
    uint8_t aes_key[AES_KEY_SIZE]; \
    uint8_t* long_state; \
    oaes_ctx* aes_ctx; \
    size_t i; \
    const size_t init_rounds = (PAGE_SIZE_CONST) / INIT_SIZE_BYTE; \
    const uint64_t offset_mask = (uint64_t)(OFFSET_MASK_CONST); \
    __m128i ax, bx; \
    if (g_yerbas_cn_scratchpad == NULL) \
        g_yerbas_cn_scratchpad = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE); \
    if (g_yerbas_cn_oaes == NULL) g_yerbas_cn_oaes = oaes_alloc(); \
    long_state = g_yerbas_cn_scratchpad; \
    aes_ctx = (oaes_ctx*)g_yerbas_cn_oaes; \
    if (long_state == NULL || aes_ctx == NULL) { \
        yerbas_cn_slow_hash_reuse_impl(input, output, (int)len, 1, \
            (PAGE_SIZE_CONST), (ITERATIONS_CONST), ((OFFSET_MASK_CONST) >> 4) + 1U); \
        return; \
    } \
    hash_process(&state.hs, (const uint8_t*)input, (int)len); \
    memcpy(text, state.init, INIT_SIZE_BYTE); \
    memcpy(aes_key, state.hs.b, AES_KEY_SIZE); \
    oaes_key_import_data(aes_ctx, aes_key, AES_KEY_SIZE); \
    for (i = 0; i < init_rounds; ++i) { \
        yerbas_aesni_pseudo_round8(text, aes_ctx->key->exp_data); \
        memcpy(&long_state[i * INIT_SIZE_BYTE], text, INIT_SIZE_BYTE); \
    } \
    ax = _mm_set_epi64x((long long)(state.hs.w[1] ^ state.hs.w[5]), \
                        (long long)(state.hs.w[0] ^ state.hs.w[4])); \
    bx = _mm_set_epi64x((long long)(state.hs.w[3] ^ state.hs.w[7]), \
                        (long long)(state.hs.w[2] ^ state.hs.w[6])); \
    { \
        const uint64_t tweak = *(const uint64_t*)((const uint8_t*)input + 35) ^ state.hs.w[24]; \
        for (i = 0; i < (ITERATIONS_CONST); ++i) { \
            const uint64_t off0 = ((uint64_t)_mm_cvtsi128_si64(ax)) & offset_mask; \
            __m128i* p0 = (__m128i*)(long_state + off0); \
            const __m128i cx = _mm_aesenc_si128(_mm_load_si128(p0), ax); \
            __m128i wr = _mm_xor_si128(bx, cx); \
            _mm_store_si128(p0, wr); \
            { \
                uint8_t* q = (uint8_t*)p0; \
                const uint8_t tmp = q[11]; \
                static const uint32_t table = 0x75310; \
                const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1; \
                q[11] = tmp ^ ((table >> index) & 0x30); \
            } \
            { \
                const uint64_t off1 = ((uint64_t)_mm_cvtsi128_si64(cx)) & offset_mask; \
                uint64_t* dst = (uint64_t*)(long_state + off1); \
                const uint64_t t0 = dst[0]; \
                const uint64_t t1 = dst[1]; \
                uint64_t hi; \
                const uint64_t lo = mul128((uint64_t)_mm_cvtsi128_si64(cx), t0, &hi); \
                uint64_t a0 = (uint64_t)_mm_cvtsi128_si64(ax) + hi; \
                uint64_t a1 = (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(ax, 8)) + lo; \
                dst[0] = a0; \
                dst[1] = a1 ^ tweak; \
                a0 ^= t0; \
                a1 ^= t1; \
                ax = _mm_set_epi64x((long long)a1, (long long)a0); \
            } \
            bx = cx; \
        } \
    } \
    memcpy(text, state.init, INIT_SIZE_BYTE); \
    oaes_key_import_data(aes_ctx, &state.hs.b[32], AES_KEY_SIZE); \
    for (i = 0; i < init_rounds; ++i) \
        yerbas_aesni_xor_pseudo_round8(text, &long_state[i * INIT_SIZE_BYTE], aes_ctx->key->exp_data); \
    memcpy(state.init, text, INIT_SIZE_BYTE); \
    hash_permutation(&state.hs); \
    extra_hashes[state.hs.b[0] & 3](&state, 200, output); \
}

/* offset mask is ((aes_rounds - 1) << 4), exactly equivalent to
 * e2i(x, aes_rounds) * AES_BLOCK_SIZE for power-of-two aes_rounds. */
YERBAS_DEFINE_CN_V1_KERNEL(yerbas_cn_dark_v1_aesni,       524288U,  131072U,  524272U)
YERBAS_DEFINE_CN_V1_KERNEL(yerbas_cn_darklite_v1_aesni,   524288U,  131072U,  262128U)
YERBAS_DEFINE_CN_V1_KERNEL(yerbas_cn_fast_v1_aesni,      2097152U,  262144U, 2097136U)
YERBAS_DEFINE_CN_V1_KERNEL(yerbas_cn_lite_v1_aesni,      1048576U,  262144U, 1048560U)
YERBAS_DEFINE_CN_V1_KERNEL(yerbas_cn_turtle_v1_aesni,     262144U,   65536U,  262128U)
YERBAS_DEFINE_CN_V1_KERNEL(yerbas_cn_turtlelite_v1_aesni, 262144U,   65536U,  131056U)
#undef YERBAS_DEFINE_CN_V1_KERNEL

YERBAS_AES_TARGET
static int yerbas_cn_dispatch_fixed_v1(const char* input,
                                       char* output,
                                       uint32_t len,
                                       uint32_t page_size,
                                       uint32_t iterations,
                                       size_t aes_rounds)
{
    if (page_size == 524288U && iterations == 131072U && aes_rounds == 32768U) {
        yerbas_cn_dark_v1_aesni(input, output, len); return 1;
    }
    if (page_size == 524288U && iterations == 131072U && aes_rounds == 16384U) {
        yerbas_cn_darklite_v1_aesni(input, output, len); return 1;
    }
    if (page_size == 2097152U && iterations == 262144U && aes_rounds == 131072U) {
        yerbas_cn_fast_v1_aesni(input, output, len); return 1;
    }
    if (page_size == 1048576U && iterations == 262144U && aes_rounds == 65536U) {
        yerbas_cn_lite_v1_aesni(input, output, len); return 1;
    }
    if (page_size == 262144U && iterations == 65536U && aes_rounds == 16384U) {
        yerbas_cn_turtle_v1_aesni(input, output, len); return 1;
    }
    if (page_size == 262144U && iterations == 65536U && aes_rounds == 8192U) {
        yerbas_cn_turtlelite_v1_aesni(input, output, len); return 1;
    }
    return 0;
}
#endif

static int g_yerbas_cn_backend_reported = 0;

void yerbas_cn_slow_hash_reuse(const char* input,
                               char* output,
                               uint32_t len,
                               int variant,
                               uint32_t page_size,
                               uint32_t iterations,
                               size_t aes_rounds)
{
    if (!g_yerbas_cn_backend_reported) {
        printf("CPU CryptoNight backend: %s | runtime-dispatch=yes | fixed-v1-kernels=yes | xmm-hotloop=yes | aes8-pipeline=yes\n",
               yerbas_cn_reuse_backend());
        g_yerbas_cn_backend_reported = 1;
    }
#if YERBAS_X86_AES_RUNTIME
    if (variant == 1 && len >= 43U && yerbas_runtime_has_aes_avx2() &&
        yerbas_cn_dispatch_fixed_v1(input, output, len, page_size, iterations, aes_rounds))
        return;
#endif
    yerbas_cn_slow_hash_reuse_impl(input, output, (int)len, variant, page_size, iterations, aes_rounds);
}

#include "cn_2way_impl.c"

int yerbas_cn_hash_pair_reference(const char* input0,
                                  const char* input1,
                                  char* output0,
                                  char* output1,
                                  uint32_t len,
                                  int variant,
                                  uint32_t page_size,
                                  uint32_t iterations,
                                  size_t aes_rounds)
{
    if (input0 == NULL || input1 == NULL || output0 == NULL || output1 == NULL) return 0;
    yerbas_cn_slow_hash_reuse(input0, output0, len, variant, page_size, iterations, aes_rounds);
    yerbas_cn_slow_hash_reuse(input1, output1, len, variant, page_size, iterations, aes_rounds);
    return 1;
}

int yerbas_cn_2way_ready(void) { return 0; }
int yerbas_cn_4way_ready(void) { return 0; }
