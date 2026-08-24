#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "slow-hash.h"
#include "oaes_lib.h"

/* Experimental CN-Fast A/B candidate.
 *
 * The generated candidate still preserves the exact pinned Core algorithm and
 * memory ordering. On GCC we ask the backend to spend more effort on register
 * renaming, instruction scheduling and loop register-pressure decisions. The
 * runtime parity/timing gate in slow_hash_reuse.c decides whether this path is
 * ever used for mining; unsupported compilers simply build the normal code.
 *
 * Note: the generated candidate currently also carries the conservative x2
 * unroll hint from the first A/B experiment. The startup report below makes
 * that explicit so benchmark results are not mistaken for a pure scheduler
 * experiment. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize ("rename-registers")
#pragma GCC optimize ("schedule-insns2")
#pragma GCC optimize ("ira-loop-pressure")
#define YERBAS_CN_CANDIDATE_MODE "sched-reg+unroll2"
#elif defined(__clang__)
#define YERBAS_CN_CANDIDATE_MODE "unroll2"
#elif defined(_MSC_VER)
#pragma optimize("t", on)
#define YERBAS_CN_CANDIDATE_MODE "msvc-speed+unroll2"
#else
#define YERBAS_CN_CANDIDATE_MODE "unroll2"
#endif

#if defined(_MSC_VER)
#define YERBAS_TLS __declspec(thread)
#else
#define YERBAS_TLS _Thread_local
#endif

#define YERBAS_CN_MAX_PAGE_SIZE 2097152u

static YERBAS_TLS uint8_t* g_candidate_scratchpad = NULL;
static YERBAS_TLS OAES_CTX* g_candidate_oaes = NULL;
static int g_candidate_mode_reported = 0;

static void* candidate_malloc(size_t requested)
{
    if (requested > YERBAS_CN_MAX_PAGE_SIZE) return NULL;
    if (g_candidate_scratchpad == NULL)
        g_candidate_scratchpad = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE);
    return g_candidate_scratchpad;
}

static void candidate_free(void* ptr)
{
    (void)ptr;
}

static OAES_CTX* candidate_oaes_alloc(void)
{
    if (g_candidate_oaes == NULL)
        g_candidate_oaes = oaes_alloc();
    return g_candidate_oaes;
}

static void candidate_oaes_free(OAES_CTX** ctx)
{
    if (ctx != NULL) *ctx = NULL;
}

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && defined(__AES__)
#include <wmmintrin.h>
#define YERBAS_CANDIDATE_AESNI 1

static int candidate_aes_single_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
    const __m128i state = _mm_loadu_si128((const __m128i*)in);
    const __m128i key = _mm_loadu_si128((const __m128i*)expanded_key);
    _mm_storeu_si128((__m128i*)out, _mm_aesenc_si128(state, key));
    return 0;
}

static int candidate_aes_pseudo_round(const uint8_t* in, uint8_t* out, const uint8_t* expanded_key)
{
    __m128i state = _mm_loadu_si128((const __m128i*)in);
    for (int round = 0; round < 10; ++round) {
        const __m128i key = _mm_loadu_si128((const __m128i*)(expanded_key + round * 16));
        state = _mm_aesenc_si128(state, key);
    }
    _mm_storeu_si128((__m128i*)out, state);
    return 0;
}
#endif

#define do_groestl_hash yerbas_candidate_do_groestl_hash
#define malloc candidate_malloc
#define free candidate_free
#if defined(_MSC_VER)
#define _malloca candidate_malloc
#define _freea candidate_free
#endif
#define oaes_alloc candidate_oaes_alloc
#define oaes_free candidate_oaes_free
#if defined(YERBAS_CANDIDATE_AESNI)
#define aesb_single_round candidate_aes_single_round
#define aesb_pseudo_round candidate_aes_pseudo_round
#endif

#include "cn_fast_candidate_impl.c"

#undef do_groestl_hash
#undef malloc
#undef free
#if defined(_MSC_VER)
#undef _malloca
#undef _freea
#endif
#undef oaes_alloc
#undef oaes_free
#if defined(YERBAS_CANDIDATE_AESNI)
#undef aesb_single_round
#undef aesb_pseudo_round
#endif

void yerbas_cn_fast_candidate(const char* input,
                              char* output,
                              uint32_t len,
                              int variant,
                              uint32_t page_size,
                              uint32_t iterations,
                              size_t aes_rounds)
{
    if (!g_candidate_mode_reported) {
        printf("[CPU CN candidate] CN-Fast | mode=%s | algorithm=pinned-core | runtime-gated=yes\n",
               YERBAS_CN_CANDIDATE_MODE);
        g_candidate_mode_reported = 1;
    }

    yerbas_cn_fast_candidate_impl(input, output, (int)len, variant,
                                  page_size, iterations, aes_rounds);
}
