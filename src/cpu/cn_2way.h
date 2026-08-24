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

/* Genuine four-lane CryptoNight kernel. Four lane-local dependency chains are
 * interleaved in one execution path. Production selection remains gated by
 * parity and measured throughput. */
int yerbas_cn_hash_quad_4way(const char* input0,
                             const char* input1,
                             const char* input2,
                             const char* input3,
                             char* output0,
                             char* output1,
                             char* output2,
                             char* output3,
                             uint32_t len,
                             int variant,
                             uint32_t page_size,
                             uint32_t iterations,
                             size_t aes_rounds);

/* Returns non-zero only after the corresponding kernel passes parity and the
 * production performance gate. */
int yerbas_cn_2way_ready(void);
int yerbas_cn_4way_ready(void);

#ifdef __cplusplus
}
#endif
