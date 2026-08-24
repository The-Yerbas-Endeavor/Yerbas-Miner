#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "slow-hash.h"
#include "oaes_lib.h"

/* Experimental CN-Fast A/B candidate.
 *
 * Keep the pinned Core loop/dataflow unchanged. This candidate isolates a
 * lower-risk AES-NI code-generation experiment: force-inline the AES round and,
 * on 64-bit x86 where malloc/_malloca provide at least 16-byte alignment, use an
 * aligned load for the scratchpad block. Key/output accesses remain unaligned.
 *
 * Runtime parity/timing selection in slow_hash_reuse.c remains the safety gate:
 * this candidate is never used for mining unless it is bit-identical and at
 * least 2% faster than the established baseline on the current machine. */
#define YERBAS_CN_CANDIDATE_MODE "aligned-aes"

#if defined(_MSC_VER)
#define YERBAS_TLS __declspec(thread)
#define YERBAS_FORCE_INLINE __forceinline
#else
#define YERBAS_TLS _Thread_local
#if defined(__GNUC__) || defined(__clang__)
#define YERBAS_FORCE_INLINE inline __attribute__((always_inline))
#else
#define YERBAS_FORCE_INLINE inline
#endif
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

static YERBAS_FORCE_INLINE int candidate_aes_single_round(const uint8_t* in,
                                                           uint8_t* out,
                                                           const uint8_t* expanded_key)
{
#if defined(__x86_64__) || defined(_M_X64)
    const __m128i state = _mm_load_si128((const __m128i*)in);
#else
    const __m128i state = _mm_loadu_si128((const __m128i*)in);
#endif
    const __m128i key = _mm_loadu_si128((const __m128i*)expanded_key);
    _mm_storeu_si128((__m128i*)out, _mm_aesenc_si128(state, key));
    return 0;
}

static YERBAS_FORCE_INLINE int candidate_aes_pseudo_round(const uint8_t* in,
                                                           uint8_t* out,
                                                           const uint8_t* expanded_key)
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

/* Include the generated pinned-Core copy for the exact helpers/macros. Its
 * generated slow-hash function is not used by the candidate below. */
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

#if defined(YERBAS_CANDIDATE_AESNI)
#define YERBAS_CANDIDATE_AES_SINGLE candidate_aes_single_round
#define YERBAS_CANDIDATE_AES_PSEUDO candidate_aes_pseudo_round
#else
#define YERBAS_CANDIDATE_AES_SINGLE aesb_single_round
#define YERBAS_CANDIDATE_AES_PSEUDO aesb_pseudo_round
#endif

static void yerbas_cn_fast_aligned_aes_impl(const char* input,
                                            char* output,
                                            int len,
                                            int variant,
                                            uint32_t page_size,
                                            uint32_t iterations,
                                            size_t aes_rounds)
{
    union cn_slow_hash_state state;
    uint8_t text[INIT_SIZE_BYTE];
    uint8_t a[AES_BLOCK_SIZE];
    uint8_t b[AES_BLOCK_SIZE * 2];
    uint8_t c[AES_BLOCK_SIZE];
    uint8_t aes_key[AES_KEY_SIZE];
    oaes_ctx* aes_ctx;
    const size_t init_rounds = page_size / INIT_SIZE_BYTE;
    uint8_t* long_state = (uint8_t*)candidate_malloc(page_size);
    size_t i, j;

    hash_process(&state.hs, (const uint8_t*)input, len);
    memcpy(text, state.init, INIT_SIZE_BYTE);
    memcpy(aes_key, state.hs.b, AES_KEY_SIZE);
    aes_ctx = (oaes_ctx*)candidate_oaes_alloc();

    VARIANT1_INIT();
    VARIANT2_INIT(b, state);

    oaes_key_import_data(aes_ctx, aes_key, AES_KEY_SIZE);
    for (i = 0; i < init_rounds; ++i) {
        for (j = 0; j < INIT_SIZE_BLK; ++j) {
            YERBAS_CANDIDATE_AES_PSEUDO(&text[AES_BLOCK_SIZE * j],
                                        &text[AES_BLOCK_SIZE * j],
                                        aes_ctx->key->exp_data);
        }
        memcpy(&long_state[i * INIT_SIZE_BYTE], text, INIT_SIZE_BYTE);
    }

    for (i = 0; i < 16; ++i) {
        a[i] = state.k[i] ^ state.k[32 + i];
        b[i] = state.k[16 + i] ^ state.k[48 + i];
    }

    /* Exact pinned-Core dependency order; no forced loop unrolling and no
     * register-state rewrite. Only the AES helper above differs. */
    for (i = 0; i < iterations; ++i) {
        j = e2i(a, aes_rounds);
        YERBAS_CANDIDATE_AES_SINGLE(&long_state[j * AES_BLOCK_SIZE], c, a);
        VARIANT2_SHUFFLE_ADD(long_state, j * AES_BLOCK_SIZE, a, b);
        xor_blocks_dst(c, b, &long_state[j * AES_BLOCK_SIZE]);
        VARIANT1_1((uint8_t*)&long_state[j * AES_BLOCK_SIZE]);

        j = e2i(c, aes_rounds);
        uint64_t* dst = (uint64_t*)&long_state[j * AES_BLOCK_SIZE];
        uint64_t t[2];
        t[0] = dst[0];
        t[1] = dst[1];

        VARIANT2_INTEGER_MATH(t, c);

        uint64_t hi;
        uint64_t lo = mul128(((uint64_t*)c)[0], t[0], &hi);

        VARIANT2_2();
        VARIANT2_SHUFFLE_ADD(long_state, j * AES_BLOCK_SIZE, a, b);

        ((uint64_t*)a)[0] += hi;
        ((uint64_t*)a)[1] += lo;

        dst[0] = ((uint64_t*)a)[0];
        dst[1] = ((uint64_t*)a)[1];

        ((uint64_t*)a)[0] ^= t[0];
        ((uint64_t*)a)[1] ^= t[1];

        VARIANT1_2((uint8_t*)&long_state[j * AES_BLOCK_SIZE]);
        copy_block(b + AES_BLOCK_SIZE, b);
        copy_block(b, c);
    }

    memcpy(text, state.init, INIT_SIZE_BYTE);
    oaes_key_import_data(aes_ctx, &state.hs.b[32], AES_KEY_SIZE);
    for (i = 0; i < init_rounds; ++i) {
        for (j = 0; j < INIT_SIZE_BLK; ++j) {
            xor_blocks(&text[j * AES_BLOCK_SIZE],
                       &long_state[i * INIT_SIZE_BYTE + j * AES_BLOCK_SIZE]);
            YERBAS_CANDIDATE_AES_PSEUDO(&text[j * AES_BLOCK_SIZE],
                                        &text[j * AES_BLOCK_SIZE],
                                        aes_ctx->key->exp_data);
        }
    }

    memcpy(state.init, text, INIT_SIZE_BYTE);
    hash_permutation(&state.hs);
    extra_hashes[state.hs.b[0] & 3](&state, 200, output);
    candidate_oaes_free((OAES_CTX**)&aes_ctx);
    candidate_free(long_state);
}

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

    yerbas_cn_fast_aligned_aes_impl(input, output, (int)len, variant,
                                    page_size, iterations, aes_rounds);
}

#undef YERBAS_CANDIDATE_AES_SINGLE
#undef YERBAS_CANDIDATE_AES_PSEUDO
#undef YERBAS_FORCE_INLINE
