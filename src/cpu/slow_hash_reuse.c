#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "slow-hash.h"
#include "oaes_lib.h"

/* Core's portable AES helpers are compiled from cryptonote/aesb.c into the
 * GhostRider reference library. Declare them before the runtime-dispatch
 * wrappers below so generic C11 builds never rely on implicit declarations. */
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

static void* yerbas_tls_cn_malloc(size_t requested)
{
    if (requested > YERBAS_CN_MAX_PAGE_SIZE) return NULL;
    if (g_yerbas_cn_scratchpad == NULL)
        g_yerbas_cn_scratchpad = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE);
    return g_yerbas_cn_scratchpad;
}

static void yerbas_tls_cn_free(void* ptr)
{
    (void)ptr;
}

static OAES_CTX* yerbas_tls_oaes_alloc(void)
{
    if (g_yerbas_cn_oaes == NULL)
        g_yerbas_cn_oaes = oaes_alloc();
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

static int yerbas_runtime_has_aes_avx2(void)
{
#if YERBAS_X86_AES_RUNTIME && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("aes") &&
           __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("sse4.2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 1);
    if ((regs[2] & (1 << 25)) == 0 || (regs[2] & (1 << 20)) == 0 ||
        (regs[2] & (1 << 27)) == 0 || (regs[2] & (1 << 28)) == 0)
        return 0;
    if ((_xgetbv(0) & 0x6) != 0x6) return 0;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0 && (regs[1] & (1 << 8)) != 0;
#else
    return 0;
#endif
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
    if (yerbas_runtime_has_aes_avx2())
        return yerbas_aesni_single_round(in, out, expanded_key);
#endif
    aesb_single_round(in, out, (uint8_t*)expanded_key);
    return 0;
}

static int yerbas_selected_pseudo_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
#if YERBAS_X86_AES_RUNTIME
    if (yerbas_runtime_has_aes_avx2())
        return yerbas_aesni_pseudo_round(in, out, expanded_key);
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
    if (yerbas_runtime_has_aes_avx2()) {
        features |= 1u << 0;
        features |= 1u << 1;
        features |= 1u << 2;
        features |= 1u << 3;
        features |= 1u << 4;
    }
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
        printf("CPU CryptoNight backend: %s | runtime-dispatch=yes\n",
               yerbas_cn_reuse_backend());
        g_yerbas_cn_backend_reported = 1;
    }

    yerbas_cn_slow_hash_reuse_impl(input, output, (int)len, variant,
                                   page_size, iterations, aes_rounds);
}

/*
 * 2-way development API.
 *
 * This is deliberately a correctness/reference baseline, not the production
 * 2-way kernel. It defines the pair interface and produces the two canonical
 * outputs using the validated 1-way implementation. The genuine 2-way kernel
 * will replace this baseline with two lane-local states/scratchpads whose hot
 * dependency loops are interleaved in one execution path. Until that kernel
 * passes parity and beats two 1-way calls, yerbas_cn_2way_ready() remains 0.
 */
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
    if (input0 == NULL || input1 == NULL || output0 == NULL || output1 == NULL)
        return 0;

    yerbas_cn_slow_hash_reuse(input0, output0, len, variant,
                              page_size, iterations, aes_rounds);
    yerbas_cn_slow_hash_reuse(input1, output1, len, variant,
                              page_size, iterations, aes_rounds);
    return 1;
}

int yerbas_cn_2way_ready(void)
{
    return 0;
}
