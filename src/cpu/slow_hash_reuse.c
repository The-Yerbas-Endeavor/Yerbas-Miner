#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

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

static double yerbas_now_ms(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static const char* yerbas_cn_variant_name(int variant)
{
    static const char* names[6] = {
        "CN-Dark", "CN-DarkLite", "CN-Fast",
        "CN-Lite", "CN-Turtle", "CN-TurtleLite"
    };
    return (variant >= 0 && variant < 6) ? names[variant] : "CN?";
}

static unsigned int g_yerbas_cn_profiled_mask = 0;
static int g_yerbas_cn_backend_reported = 0;

/*
 * Compile a second copy of the pristine Core CryptoNight implementation with
 * only production-specific resource/AES substitutions. The untouched pinned
 * Core implementation remains linked beside this candidate for parity checks.
 */
#define cn_slow_hash yerbas_cn_slow_hash_reuse_impl
#define cn_fast_hash yerbas_cn_fast_hash_reuse_impl
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

#undef cn_slow_hash
#undef cn_fast_hash
#undef do_groestl_hash
#undef malloc
#undef free
#undef oaes_alloc
#undef oaes_free
#if YERBAS_CN_AESNI
#undef aesb_single_round
#undef aesb_pseudo_round
#endif

void yerbas_cn_slow_hash_reuse(const char* input,
                               char* output,
                               uint32_t len,
                               int variant,
                               uint32_t page_size,
                               uint32_t iterations,
                               size_t aes_rounds)
{
    if (!g_yerbas_cn_backend_reported) {
        const unsigned int features = yerbas_cn_reuse_compile_features();
        printf("CPU CryptoNight backend: %s | compile AES=%s AVX=%s AVX2=%s BMI2=%s SSE4.2=%s\n",
               yerbas_cn_reuse_backend(),
               (features & (1u << 0)) ? "yes" : "no",
               (features & (1u << 1)) ? "yes" : "no",
               (features & (1u << 2)) ? "yes" : "no",
               (features & (1u << 3)) ? "yes" : "no",
               (features & (1u << 4)) ? "yes" : "no");
        g_yerbas_cn_backend_reported = 1;
    }

    const unsigned int bit = (variant >= 0 && variant < 6) ? (1u << variant) : 0u;
    const int profile = bit != 0u && (g_yerbas_cn_profiled_mask & bit) == 0u;
    const double start_ms = profile ? yerbas_now_ms() : 0.0;

    yerbas_cn_slow_hash_reuse_impl(input, output, (int)len, variant,
                                   page_size, iterations, aes_rounds);

    if (profile) {
        const double elapsed_ms = yerbas_now_ms() - start_ms;
        g_yerbas_cn_profiled_mask |= bit;
        printf("[CPU CN profile] %s | backend=%s | page=%u KiB | iterations=%u | elapsed=%.3f ms\n",
               yerbas_cn_variant_name(variant), yerbas_cn_reuse_backend(),
               page_size / 1024u, iterations, elapsed_ms);
    }
}
