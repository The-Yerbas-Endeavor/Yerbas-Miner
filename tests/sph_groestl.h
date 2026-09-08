#pragma once

/* Minimal ABI-compatible interface for the pinned Yerbas Core sph_groestl.h
 * used by tests/cuda_keccak_validation.cpp. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t sph_u32;
typedef uint64_t sph_u64;

typedef struct {
    unsigned char buf[64];
    size_t ptr;
    union {
        sph_u64 wide[8];
        sph_u32 narrow[16];
    } state;
#if UINTPTR_MAX > 0xffffffffU
    sph_u64 count;
#else
    sph_u32 count_high, count_low;
#endif
} sph_groestl_small_context;

typedef sph_groestl_small_context sph_groestl256_context;

void sph_groestl256_init(void *cc);
void sph_groestl256(void *cc, const void *data, size_t len);
void sph_groestl256_close(void *cc, void *dst);

#ifdef __cplusplus
}
#endif
