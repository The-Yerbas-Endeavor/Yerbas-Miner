#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "slow-hash.h"
#include "oaes_lib.h"

#if defined(_MSC_VER)
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

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && defined(__AES__)
#include <wmmintrin.h>
#define YERBAS_CN_AESNI 1

static int yerbas_aesni_single_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
    const __m128i state = _mm_loadu_si128((const __m128i*)in);
    const __m128i key = _mm_loadu_si128((const __m128i*)expanded_key);
    const __m128i result = _mm_aesenc_si128(state, key);
    _mm_storeu_si128((__m128i*)out, result);
    return 0;
}

static int yerbas_aesni_pseudo_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
    __m128i state = _mm_loadu_si128((const __m128i*)in);
    for (int round = 0; round < 10; ++round) {
        const __m128i key = _mm_loadu_si128((const __m128i*)(expanded_key + round * 16));
        state = _mm_aesenc_si128(state, key);
    }
    _mm_storeu_si128((__m128i*)out, state);
    return 0;
}
#else
#define YERBAS_CN_AESNI 0
#endif

const char* yerbas_cn_reuse_backend(void)
{
#if YERBAS_CN_AESNI
    return "aes-ni";
#else
    return "portable-table-aes";
#endif
}

unsigned int yerbas_cn_reuse_compile_features(void)
{
    unsigned int features = 0;
#ifdef __AES__
    features |= 1u << 0;
#endif
#ifdef __AVX__
    features |= 1u << 1;
#endif
#ifdef __AVX2__
    features |= 1u << 2;
#endif
#ifdef __BMI2__
    features |= 1u << 3;
#endif
#ifdef __SSE4_2__
    features |= 1u << 4;
#endif
    return features;
}

/*
 * Compile a second copy of the pristine Core CryptoNight implementation with
 * only production-specific resource/AES substitutions. The untouched pinned
 * Core implementation remains linked beside this candidate for parity checks.
 */
#define cn_slow_hash yerbas_cn_slow_hash_reuse
#define cn_fast_hash yerbas_cn_fast_hash_reuse
#define do_groestl_hash yerbas_reuse_do_groestl_hash
#define malloc yerbas_tls_cn_malloc
#define free yerbas_tls_cn_free
#define oaes_alloc yerbas_tls_oaes_alloc
#define oaes_free yerbas_tls_oaes_free
#if YERBAS_CN_AESNI
#define aesb_single_round yerbas_aesni_single_round
#define aesb_pseudo_round yerbas_aesni_pseudo_round
#endif

#include "slow-hash.c"
