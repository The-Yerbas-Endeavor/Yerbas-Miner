#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Two validated 1-way hashes used as the correctness/timing baseline. */
int yerbas_cn_hash_pair_reference(const char* input0,
                                  const char* input1,
                                  char* output0,
                                  char* output1,
                                  uint32_t len,
                                  int variant,
                                  uint32_t page_size,
                                  uint32_t iterations,
                                  size_t aes_rounds);

/* Genuine two-lane CryptoNight kernel. Both lanes have independent state and
 * scratchpads; their dependency-heavy loops are interleaved in one execution
 * path. This function remains experimental until yerbas_cn_2way_ready() is
 * non-zero after parity/performance validation. */
int yerbas_cn_hash_pair_2way(const char* input0,
                             const char* input1,
                             char* output0,
                             char* output1,
                             uint32_t len,
                             int variant,
                             uint32_t page_size,
                             uint32_t iterations,
                             size_t aes_rounds);

/* Returns non-zero only after the 2-way kernel passes parity and the production
 * performance gate. */
int yerbas_cn_2way_ready(void);

/* 4-way development status. The next layer will compose/interleave four lane
 * states after the 2-way kernel is validated. */
int yerbas_cn_4way_ready(void);

#ifdef __cplusplus
}
#endif
