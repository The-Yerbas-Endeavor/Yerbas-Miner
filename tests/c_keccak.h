#pragma once

/* Minimal interface from the pinned Yerbas Core cryptonote/c_keccak.h. */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int keccak(const uint8_t *in, int inlen, uint8_t *md, int mdlen);
void keccakf(uint64_t st[25], int norounds);
void keccak1600(const uint8_t *in, int inlen, uint8_t *md);

#ifdef __cplusplus
}
#endif
