#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

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

/*
 * Compile a second copy of the pristine Core CryptoNight implementation with
 * only resource-lifetime changes. All algorithm code remains byte-for-byte the
 * pinned Core source. Exported symbols are renamed so the untouched reference
 * implementation can remain linked beside this production candidate.
 */
#define cn_slow_hash yerbas_cn_slow_hash_reuse
#define cn_fast_hash yerbas_cn_fast_hash_reuse
#define do_groestl_hash yerbas_reuse_do_groestl_hash
#define malloc yerbas_tls_cn_malloc
#define free yerbas_tls_cn_free
#define oaes_alloc yerbas_tls_oaes_alloc
#define oaes_free yerbas_tls_oaes_free

#include "slow-hash.c"
