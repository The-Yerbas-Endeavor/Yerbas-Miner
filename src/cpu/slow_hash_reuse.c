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

static int yerbas_cn_profile_variant(uint32_t page_size,
                                     uint32_t iterations,
                                     size_t aes_rounds)
{
    if (page_size == 524288u && iterations == 131072u && aes_rounds == 32768u) return 0;
    if (page_size == 524288u && iterations == 131072u && aes_rounds == 16384u) return 1;
    if (page_size == 2097152u && iterations == 262144u && aes_rounds == 131072u) return 2;
    if (page_size == 1048576u && iterations == 262144u && aes_rounds == 65536u) return 3;
    if (page_size == 262144u && iterations == 65536u && aes_rounds == 16384u) return 4;
    if (page_size == 262144u && iterations == 65536u && aes_rounds == 8192u) return 5;
    return -1;
}

static unsigned int g_yerbas_cn_profiled_mask = 0;
static int g_yerbas_cn_backend_reported = 0;
static int g_yerbas_cn_fast_phase_reported = 0;
/* 0=untested, 1=baseline selected, 2=unroll2 candidate selected. */
static int g_yerbas_cn_fast_ab_state = 0;

void yerbas_cn_fast_candidate(const char* input,
                              char* output,
                              uint32_t len,
                              int variant,
                              uint32_t page_size,
                              uint32_t iterations,
                              size_t aes_rounds);

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
#if defined(_MSC_VER)
#undef _malloca
#undef _freea
#endif
#undef oaes_alloc
#undef oaes_free
#if YERBAS_CN_AESNI
#undef aesb_single_round
#undef aesb_pseudo_round
#endif

/* Backend-neutral AES helpers for profiling. The production slow-hash copy above
 * still selects its backend at compile time; these wrappers merely let the
 * diagnostic phase probe compile and follow the same available backend in both
 * native and portable release builds. */
static void yerbas_profile_aes_single_round(const uint8_t* in,
                                            uint8_t* out,
                                            const uint8_t* expanded_key)
{
#if YERBAS_CN_AESNI
    (void)yerbas_aesni_single_round(in, out, expanded_key);
#else
    aesb_single_round(in, out, expanded_key);
#endif
}

static void yerbas_profile_aes_pseudo_round(const uint8_t* in,
                                            uint8_t* out,
                                            const uint8_t* expanded_key)
{
#if YERBAS_CN_AESNI
    (void)yerbas_aesni_pseudo_round(in, out, expanded_key);
#else
    aesb_pseudo_round(in, out, expanded_key);
#endif
}

static void yerbas_cn_fast_ab_tune(const char* input,
                                   const char* known_good_output,
                                   uint32_t len,
                                   int variant,
                                   uint32_t page_size,
                                   uint32_t iterations,
                                   size_t aes_rounds)
{
    if (g_yerbas_cn_fast_ab_state != 0) return;

    char baseline_output[HASH_SIZE];
    char candidate_output[HASH_SIZE];
    double baseline_ms = 0.0;
    double candidate_ms = 0.0;

    for (int round = 0; round < 2; ++round) {
        double start = yerbas_now_ms();
        yerbas_cn_slow_hash_reuse_impl(input, baseline_output, (int)len, variant,
                                       page_size, iterations, aes_rounds);
        baseline_ms += yerbas_now_ms() - start;

        start = yerbas_now_ms();
        yerbas_cn_fast_candidate(input, candidate_output, len, variant,
                                 page_size, iterations, aes_rounds);
        candidate_ms += yerbas_now_ms() - start;
    }

    baseline_ms *= 0.5;
    candidate_ms *= 0.5;

    const int baseline_parity = memcmp(baseline_output, known_good_output, HASH_SIZE) == 0;
    const int candidate_parity = memcmp(candidate_output, known_good_output, HASH_SIZE) == 0;
    const int parity_ok = baseline_parity && candidate_parity;
    const int faster = candidate_ms > 0.0 && candidate_ms <= baseline_ms * 0.98;

    g_yerbas_cn_fast_ab_state = (parity_ok && faster) ? 2 : 1;

    printf("[CPU CN A/B] CN-Fast | baseline=%.3f ms | candidate=%.3f ms | parity=%s | selected=%s\n",
           baseline_ms, candidate_ms, parity_ok ? "PASS" : "FAIL",
           g_yerbas_cn_fast_ab_state == 2 ? "candidate" : "baseline");
}

static void yerbas_cn_fast_phase_probe(const char* input,
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
    uint8_t output[HASH_SIZE];
    oaes_ctx* aes_ctx;
    size_t init_rounds = page_size / INIT_SIZE_BYTE;
    uint8_t* long_state = (uint8_t*)yerbas_tls_cn_malloc(page_size);
    size_t i, j;

    const double total_start = yerbas_now_ms();
    hash_process(&state.hs, (const uint8_t*)input, len);
    memcpy(text, state.init, INIT_SIZE_BYTE);
    memcpy(aes_key, state.hs.b, AES_KEY_SIZE);
    aes_ctx = (oaes_ctx*)yerbas_tls_oaes_alloc();
    VARIANT1_INIT();
    VARIANT2_INIT(b, state);
    oaes_key_import_data(aes_ctx, aes_key, AES_KEY_SIZE);
    for (i = 0; i < init_rounds; ++i) {
        for (j = 0; j < INIT_SIZE_BLK; ++j) {
            yerbas_profile_aes_pseudo_round(&text[AES_BLOCK_SIZE * j],
                                            &text[AES_BLOCK_SIZE * j],
                                            aes_ctx->key->exp_data);
        }
        memcpy(&long_state[i * INIT_SIZE_BYTE], text, INIT_SIZE_BYTE);
    }
    for (i = 0; i < 16; ++i) {
        a[i] = state.k[i] ^ state.k[32 + i];
        b[i] = state.k[16 + i] ^ state.k[48 + i];
    }
    const double setup_end = yerbas_now_ms();

    for (i = 0; i < iterations; ++i) {
        j = e2i(a, aes_rounds);
        yerbas_profile_aes_single_round(&long_state[j * AES_BLOCK_SIZE], c, a);
        VARIANT2_SHUFFLE_ADD(long_state, j * AES_BLOCK_SIZE, a, b);
        xor_blocks_dst(c, b, &long_state[j * AES_BLOCK_SIZE]);
        VARIANT1_1((uint8_t*)&long_state[j * AES_BLOCK_SIZE]);
        j = e2i(c, aes_rounds);
        uint64_t* dst = (uint64_t*)&long_state[j * AES_BLOCK_SIZE];
        uint64_t t[2] = {dst[0], dst[1]};
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
    const double loop_end = yerbas_now_ms();

    memcpy(text, state.init, INIT_SIZE_BYTE);
    oaes_key_import_data(aes_ctx, &state.hs.b[32], AES_KEY_SIZE);
    for (i = 0; i < init_rounds; ++i) {
        for (j = 0; j < INIT_SIZE_BLK; ++j) {
            xor_blocks(&text[j * AES_BLOCK_SIZE],
                       &long_state[i * INIT_SIZE_BYTE + j * AES_BLOCK_SIZE]);
            yerbas_profile_aes_pseudo_round(&text[j * AES_BLOCK_SIZE],
                                            &text[j * AES_BLOCK_SIZE],
                                            aes_ctx->key->exp_data);
        }
    }
    memcpy(state.init, text, INIT_SIZE_BYTE);
    hash_permutation(&state.hs);
    extra_hashes[state.hs.b[0] & 3](&state, 200, (char*)output);
    const double final_end = yerbas_now_ms();

    const double setup_ms = setup_end - total_start;
    const double loop_ms = loop_end - setup_end;
    const double final_ms = final_end - loop_end;
    const double total_ms = final_end - total_start;
    printf("[CPU CN phase] CN-Fast | backend=%s | setup=%.3f ms | loop=%.3f ms | final=%.3f ms | total=%.3f ms | loop=%.1f%%\n",
           yerbas_cn_reuse_backend(), setup_ms, loop_ms, final_ms, total_ms,
           total_ms > 0.0 ? (loop_ms * 100.0 / total_ms) : 0.0);
}

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

    const int profile_variant = yerbas_cn_profile_variant(page_size, iterations, aes_rounds);
    const double start = yerbas_now_ms();

    yerbas_cn_slow_hash_reuse_impl(input, output, (int)len, variant,
                                   page_size, iterations, aes_rounds);

    const double elapsed = yerbas_now_ms() - start;
    if (profile_variant >= 0 && (g_yerbas_cn_profiled_mask & (1u << profile_variant)) == 0) {
        printf("[CPU CN profile] %s | backend=%s | page=%u KiB | iterations=%u | elapsed=%.3f ms\n",
               yerbas_cn_variant_name(profile_variant), yerbas_cn_reuse_backend(),
               page_size / 1024u, iterations, elapsed);
        g_yerbas_cn_profiled_mask |= 1u << profile_variant;
    }

    if (profile_variant == 2 && g_yerbas_cn_fast_ab_state == 0) {
        yerbas_cn_fast_ab_tune(input, output, len, variant, page_size, iterations, aes_rounds);
    }
    if (profile_variant == 2 && !g_yerbas_cn_fast_phase_reported) {
        yerbas_cn_fast_phase_probe(input, (int)len, variant, page_size, iterations, aes_rounds);
        g_yerbas_cn_fast_phase_reported = 1;
    }
}
