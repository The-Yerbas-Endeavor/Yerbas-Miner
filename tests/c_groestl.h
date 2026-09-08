#pragma once

/* Minimal interface from the pinned Yerbas Core cryptonote/c_groestl.h. */
#include <stdint.h>
#include "slow-hash.h"

#ifdef __cplusplus
extern "C" {
#endif

void groestl(const BitSequence *input, DataLength bit_length, BitSequence *output);

#ifdef __cplusplus
}
#endif
