/* Genuine two-lane CryptoNight execution path.
 *
 * This file is included at the end of slow_hash_reuse.c so it can reuse the
 * pinned Core CryptoNight helpers and the runtime-selected AES implementation.
 * Each lane owns a separate scratchpad and state. The dependency-heavy memory
 * loop is interleaved lane 0 / lane 1 every iteration to expose independent
 * work to the CPU while preserving the exact variant-1 algorithm per lane.
 */

static YERBAS_TLS uint8_t* g_yerbas_cn_2way_scratchpad0 = NULL;
static YERBAS_TLS uint8_t* g_yerbas_cn_2way_scratchpad1 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_2way_oaes0 = NULL;
static YERBAS_TLS OAES_CTX* g_yerbas_cn_2way_oaes1 = NULL;

static int yerbas_cn_2way_resources(void)
{
    if (g_yerbas_cn_2way_scratchpad0 == NULL)
        g_yerbas_cn_2way_scratchpad0 = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE);
    if (g_yerbas_cn_2way_scratchpad1 == NULL)
        g_yerbas_cn_2way_scratchpad1 = (uint8_t*)malloc(YERBAS_CN_MAX_PAGE_SIZE);
    if (g_yerbas_cn_2way_oaes0 == NULL)
        g_yerbas_cn_2way_oaes0 = oaes_alloc();
    if (g_yerbas_cn_2way_oaes1 == NULL)
        g_yerbas_cn_2way_oaes1 = oaes_alloc();
    return g_yerbas_cn_2way_scratchpad0 != NULL &&
           g_yerbas_cn_2way_scratchpad1 != NULL &&
           g_yerbas_cn_2way_oaes0 != NULL &&
           g_yerbas_cn_2way_oaes1 != NULL;
}

static void yerbas_cn_lane_setup(const char* input,
                                 int len,
                                 union cn_slow_hash_state* state,
                                 uint8_t text[INIT_SIZE_BYTE],
                                 uint8_t a[AES_BLOCK_SIZE],
                                 uint8_t b[AES_BLOCK_SIZE * 2],
                                 uint8_t* long_state,
                                 OAES_CTX* aes_ctx,
                                 uint32_t page_size)
{
    uint8_t aes_key[AES_KEY_SIZE];
    const size_t init_rounds = page_size / INIT_SIZE_BYTE;
    size_t i, j;

    hash_process(&state->hs, (const uint8_t*)input, len);
    memcpy(text, state->init, INIT_SIZE_BYTE);
    memcpy(aes_key, state->hs.b, AES_KEY_SIZE);
    oaes_key_import_data((oaes_ctx*)aes_ctx, aes_key, AES_KEY_SIZE);

    for (i = 0; i < init_rounds; ++i) {
        for (j = 0; j < INIT_SIZE_BLK; ++j) {
            yerbas_selected_pseudo_round(&text[AES_BLOCK_SIZE * j],
                                         &text[AES_BLOCK_SIZE * j],
                                         ((oaes_ctx*)aes_ctx)->key->exp_data);
        }
        memcpy(&long_state[i * INIT_SIZE_BYTE], text, INIT_SIZE_BYTE);
    }

    for (i = 0; i < 16; ++i) {
        a[i] = state->k[i] ^ state->k[32 + i];
        b[i] = state->k[16 + i] ^ state->k[48 + i];
    }
    memset(b + AES_BLOCK_SIZE, 0, AES_BLOCK_SIZE);
}

static void yerbas_cn_lane_loop_step(uint8_t a[AES_BLOCK_SIZE],
                                     uint8_t b[AES_BLOCK_SIZE * 2],
                                     uint8_t c[AES_BLOCK_SIZE],
                                     uint8_t* long_state,
                                     size_t aes_rounds,
                                     uint64_t tweak1_2)
{
    size_t j = e2i(a, aes_rounds);
    uint64_t* dst;
    uint64_t t[2];
    uint64_t hi, lo;

    yerbas_selected_single_round(&long_state[j * AES_BLOCK_SIZE], c, a);
    xor_blocks_dst(c, b, &long_state[j * AES_BLOCK_SIZE]);
    {
        const uint8_t tmp = long_state[j * AES_BLOCK_SIZE + 11];
        static const uint32_t table = 0x75310;
        const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1;
        long_state[j * AES_BLOCK_SIZE + 11] = tmp ^ ((table >> index) & 0x30);
    }

    j = e2i(c, aes_rounds);
    dst = (uint64_t*)&long_state[j * AES_BLOCK_SIZE];
    t[0] = dst[0];
    t[1] = dst[1];
    lo = mul128(((uint64_t*)c)[0], t[0], &hi);

    ((uint64_t*)a)[0] += hi;
    ((uint64_t*)a)[1] += lo;
    dst[0] = ((uint64_t*)a)[0];
    dst[1] = ((uint64_t*)a)[1];
    ((uint64_t*)a)[0] ^= t[0];
    ((uint64_t*)a)[1] ^= t[1];
    ((uint64_t*)&long_state[j * AES_BLOCK_SIZE])[1] ^= tweak1_2;

    copy_block(b + AES_BLOCK_SIZE, b);
    copy_block(b, c);
}

static void yerbas_cn_lane_finish(union cn_slow_hash_state* state,
                                  uint8_t text[INIT_SIZE_BYTE],
                                  uint8_t* long_state,
                                  OAES_CTX* aes_ctx,
                                  uint32_t page_size,
                                  char* output)
{
    const size_t init_rounds = page_size / INIT_SIZE_BYTE;
    size_t i, j;

    memcpy(text, state->init, INIT_SIZE_BYTE);
    oaes_key_import_data((oaes_ctx*)aes_ctx, &state->hs.b[32], AES_KEY_SIZE);
    for (i = 0; i < init_rounds; ++i) {
        for (j = 0; j < INIT_SIZE_BLK; ++j) {
            xor_blocks(&text[j * AES_BLOCK_SIZE],
                       &long_state[i * INIT_SIZE_BYTE + j * AES_BLOCK_SIZE]);
            yerbas_selected_pseudo_round(&text[j * AES_BLOCK_SIZE],
                                         &text[j * AES_BLOCK_SIZE],
                                         ((oaes_ctx*)aes_ctx)->key->exp_data);
        }
    }
    memcpy(state->init, text, INIT_SIZE_BYTE);
    hash_permutation(&state->hs);
    extra_hashes[state->hs.b[0] & 3](state, 200, output);
}

int yerbas_cn_hash_pair_2way(const char* input0,
                             const char* input1,
                             char* output0,
                             char* output1,
                             uint32_t len,
                             int variant,
                             uint32_t page_size,
                             uint32_t iterations,
                             size_t aes_rounds)
{
    union cn_slow_hash_state state0, state1;
    uint8_t text0[INIT_SIZE_BYTE], text1[INIT_SIZE_BYTE];
    uint8_t a0[AES_BLOCK_SIZE], a1[AES_BLOCK_SIZE];
    uint8_t b0[AES_BLOCK_SIZE * 2], b1[AES_BLOCK_SIZE * 2];
    uint8_t c0[AES_BLOCK_SIZE], c1[AES_BLOCK_SIZE];
    uint64_t tweak0, tweak1;
    uint32_t i;

    if (input0 == NULL || input1 == NULL || output0 == NULL || output1 == NULL)
        return 0;
    /* Yerbas GhostRider currently invokes CryptoNight variant 1 for all six
     * profiles. Keep production eligibility strict until other variants have
     * their own parity coverage. */
    if (variant != 1 || len < 43U || page_size > YERBAS_CN_MAX_PAGE_SIZE)
        return 0;
    if (!yerbas_cn_2way_resources()) return 0;

    yerbas_cn_lane_setup(input0, (int)len, &state0, text0, a0, b0,
                         g_yerbas_cn_2way_scratchpad0,
                         g_yerbas_cn_2way_oaes0, page_size);
    yerbas_cn_lane_setup(input1, (int)len, &state1, text1, a1, b1,
                         g_yerbas_cn_2way_scratchpad1,
                         g_yerbas_cn_2way_oaes1, page_size);

    tweak0 = *(const uint64_t*)((const uint8_t*)input0 + 35) ^ state0.hs.w[24];
    tweak1 = *(const uint64_t*)((const uint8_t*)input1 + 35) ^ state1.hs.w[24];

    for (i = 0; i < iterations; ++i) {
        yerbas_cn_lane_loop_step(a0, b0, c0, g_yerbas_cn_2way_scratchpad0,
                                 aes_rounds, tweak0);
        yerbas_cn_lane_loop_step(a1, b1, c1, g_yerbas_cn_2way_scratchpad1,
                                 aes_rounds, tweak1);
    }

    yerbas_cn_lane_finish(&state0, text0, g_yerbas_cn_2way_scratchpad0,
                          g_yerbas_cn_2way_oaes0, page_size, output0);
    yerbas_cn_lane_finish(&state1, text1, g_yerbas_cn_2way_scratchpad1,
                          g_yerbas_cn_2way_oaes1, page_size, output1);
    return 1;
}
