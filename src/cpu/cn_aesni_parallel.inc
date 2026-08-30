/* Eight-block AES-NI helpers for CryptoNight setup/final phases.
 *
 * The 128-byte CryptoNight text buffer contains eight independent AES blocks
 * that all use the same expanded key. Processing one complete 10-round block
 * at a time serializes AES latency. These helpers instead keep all eight blocks
 * live in registers and issue each AES round across the whole group, exposing
 * instruction-level parallelism without changing any CryptoNight semantics.
 */

YERBAS_AES_TARGET
static inline void yerbas_aesni_pseudo_round8(uint8_t text[INIT_SIZE_BYTE],
                                              const uint8_t* expanded_key)
{
    __m128i s0 = _mm_loadu_si128((const __m128i*)(text + 0 * AES_BLOCK_SIZE));
    __m128i s1 = _mm_loadu_si128((const __m128i*)(text + 1 * AES_BLOCK_SIZE));
    __m128i s2 = _mm_loadu_si128((const __m128i*)(text + 2 * AES_BLOCK_SIZE));
    __m128i s3 = _mm_loadu_si128((const __m128i*)(text + 3 * AES_BLOCK_SIZE));
    __m128i s4 = _mm_loadu_si128((const __m128i*)(text + 4 * AES_BLOCK_SIZE));
    __m128i s5 = _mm_loadu_si128((const __m128i*)(text + 5 * AES_BLOCK_SIZE));
    __m128i s6 = _mm_loadu_si128((const __m128i*)(text + 6 * AES_BLOCK_SIZE));
    __m128i s7 = _mm_loadu_si128((const __m128i*)(text + 7 * AES_BLOCK_SIZE));

    for (int round = 0; round < 10; ++round) {
        const __m128i key = _mm_loadu_si128((const __m128i*)(expanded_key + round * AES_BLOCK_SIZE));
        s0 = _mm_aesenc_si128(s0, key);
        s1 = _mm_aesenc_si128(s1, key);
        s2 = _mm_aesenc_si128(s2, key);
        s3 = _mm_aesenc_si128(s3, key);
        s4 = _mm_aesenc_si128(s4, key);
        s5 = _mm_aesenc_si128(s5, key);
        s6 = _mm_aesenc_si128(s6, key);
        s7 = _mm_aesenc_si128(s7, key);
    }

    _mm_storeu_si128((__m128i*)(text + 0 * AES_BLOCK_SIZE), s0);
    _mm_storeu_si128((__m128i*)(text + 1 * AES_BLOCK_SIZE), s1);
    _mm_storeu_si128((__m128i*)(text + 2 * AES_BLOCK_SIZE), s2);
    _mm_storeu_si128((__m128i*)(text + 3 * AES_BLOCK_SIZE), s3);
    _mm_storeu_si128((__m128i*)(text + 4 * AES_BLOCK_SIZE), s4);
    _mm_storeu_si128((__m128i*)(text + 5 * AES_BLOCK_SIZE), s5);
    _mm_storeu_si128((__m128i*)(text + 6 * AES_BLOCK_SIZE), s6);
    _mm_storeu_si128((__m128i*)(text + 7 * AES_BLOCK_SIZE), s7);
}

YERBAS_AES_TARGET
static inline void yerbas_aesni_xor_pseudo_round8(uint8_t text[INIT_SIZE_BYTE],
                                                  const uint8_t scratch[INIT_SIZE_BYTE],
                                                  const uint8_t* expanded_key)
{
    __m128i s0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 0 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 0 * AES_BLOCK_SIZE)));
    __m128i s1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 1 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 1 * AES_BLOCK_SIZE)));
    __m128i s2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 2 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 2 * AES_BLOCK_SIZE)));
    __m128i s3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 3 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 3 * AES_BLOCK_SIZE)));
    __m128i s4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 4 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 4 * AES_BLOCK_SIZE)));
    __m128i s5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 5 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 5 * AES_BLOCK_SIZE)));
    __m128i s6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 6 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 6 * AES_BLOCK_SIZE)));
    __m128i s7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(text + 7 * AES_BLOCK_SIZE)), _mm_loadu_si128((const __m128i*)(scratch + 7 * AES_BLOCK_SIZE)));

    for (int round = 0; round < 10; ++round) {
        const __m128i key = _mm_loadu_si128((const __m128i*)(expanded_key + round * AES_BLOCK_SIZE));
        s0 = _mm_aesenc_si128(s0, key);
        s1 = _mm_aesenc_si128(s1, key);
        s2 = _mm_aesenc_si128(s2, key);
        s3 = _mm_aesenc_si128(s3, key);
        s4 = _mm_aesenc_si128(s4, key);
        s5 = _mm_aesenc_si128(s5, key);
        s6 = _mm_aesenc_si128(s6, key);
        s7 = _mm_aesenc_si128(s7, key);
    }

    _mm_storeu_si128((__m128i*)(text + 0 * AES_BLOCK_SIZE), s0);
    _mm_storeu_si128((__m128i*)(text + 1 * AES_BLOCK_SIZE), s1);
    _mm_storeu_si128((__m128i*)(text + 2 * AES_BLOCK_SIZE), s2);
    _mm_storeu_si128((__m128i*)(text + 3 * AES_BLOCK_SIZE), s3);
    _mm_storeu_si128((__m128i*)(text + 4 * AES_BLOCK_SIZE), s4);
    _mm_storeu_si128((__m128i*)(text + 5 * AES_BLOCK_SIZE), s5);
    _mm_storeu_si128((__m128i*)(text + 6 * AES_BLOCK_SIZE), s6);
    _mm_storeu_si128((__m128i*)(text + 7 * AES_BLOCK_SIZE), s7);
}
