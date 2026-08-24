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

/* Genuine two-lane CryptoNight kernel. */
int yerbas_cn_hash_pair_2way(const char* input0,
                             const char* input1,
                             char* output0,
                             char* output1,
                             uint32_t len,
                             int variant,
                             uint32_t page_size,
                             uint32_t iterations,
                             size_t aes_rounds);

/* Genuine four-lane CryptoNight kernel. */
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

/* These remain false until the startup parity/performance gate explicitly
 * qualifies the corresponding kernel for production mining. */
int yerbas_cn_2way_ready(void);
int yerbas_cn_4way_ready(void);

#ifdef __cplusplus
}
#endif
