#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Reference pair interface for CryptoNight 2-way development.
 *
 * yerbas_cn_hash_pair_reference() intentionally executes two validated 1-way
 * hashes and exists only as the parity/timing baseline for the future genuine
 * 2-way kernel. Production code must not treat it as a 2-way optimization.
 */
int yerbas_cn_hash_pair_reference(const char* input0,
                                  const char* input1,
                                  char* output0,
                                  char* output1,
                                  uint32_t len,
                                  int variant,
                                  uint32_t page_size,
                                  uint32_t iterations,
                                  size_t aes_rounds);

/* Returns non-zero only after an interleaved two-lane kernel is implemented,
 * passes parity against the pair reference, and is allowed for production. */
int yerbas_cn_2way_ready(void);

#ifdef __cplusplus
}
#endif
