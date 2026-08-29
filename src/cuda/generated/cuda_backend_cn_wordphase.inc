// Register-word CryptoNight setup/final candidates.
//
// These kernels preserve the proven 8-lane/hash shared-key layout, but keep each
// 16-byte AES chain as four uint32_t registers for all ten pseudo-rounds. This
// avoids repeatedly materializing/indexing a byte-local text[16] array. The AES
// T-tables and expanded 160-byte key remain shared per 8-lane subgroup.

namespace {

__device__ __forceinline__ std::uint32_t cn_word_byte(std::uint32_t w, int shift)
{
    return (w >> shift) & 0xffU;
}

__device__ __forceinline__ void cn_aes_round_words(std::uint32_t& w0,
                                                   std::uint32_t& w1,
                                                   std::uint32_t& w2,
                                                   std::uint32_t& w3,
                                                   const std::uint8_t* key,
                                                   const std::uint32_t* tables)
{
    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + 256;
    const std::uint32_t* t2 = tables + 512;
    const std::uint32_t* t3 = tables + 768;

    const std::uint32_t o0 =
        t0[cn_word_byte(w0, 0)]  ^ t1[cn_word_byte(w1, 8)]  ^
        t2[cn_word_byte(w2, 16)] ^ t3[cn_word_byte(w3, 24)] ^
        cryptonight::load32le_aligned(key);
    const std::uint32_t o1 =
        t0[cn_word_byte(w1, 0)]  ^ t1[cn_word_byte(w2, 8)]  ^
        t2[cn_word_byte(w3, 16)] ^ t3[cn_word_byte(w0, 24)] ^
        cryptonight::load32le_aligned(key + 4);
    const std::uint32_t o2 =
        t0[cn_word_byte(w2, 0)]  ^ t1[cn_word_byte(w3, 8)]  ^
        t2[cn_word_byte(w0, 16)] ^ t3[cn_word_byte(w1, 24)] ^
        cryptonight::load32le_aligned(key + 8);
    const std::uint32_t o3 =
        t0[cn_word_byte(w3, 0)]  ^ t1[cn_word_byte(w0, 8)]  ^
        t2[cn_word_byte(w1, 16)] ^ t3[cn_word_byte(w2, 24)] ^
        cryptonight::load32le_aligned(key + 12);
    w0 = o0; w1 = o1; w2 = o2; w3 = o3;
}

__device__ __forceinline__ void cn_aes_pseudo_words(std::uint32_t& w0,
                                                    std::uint32_t& w1,
                                                    std::uint32_t& w2,
                                                    std::uint32_t& w3,
                                                    const std::uint8_t* expanded,
                                                    const std::uint32_t* tables)
{
    #pragma unroll
    for (int round = 0; round < 10; ++round)
        cn_aes_round_words(w0, w1, w2, w3, expanded + round * 16, tables);
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_setup_stage_word8(std::uint8_t* states,
                                              std::size_t count,
                                              std::uint8_t* scratchpads,
                                              cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    __shared__ __align__(16) std::uint8_t shared_keys[16][160];
    cryptonight::init_shared_aes_ttables(aes_tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int lane = warp_lane & 7U;
    const unsigned int base = warp_lane & ~7U;
    const unsigned int group = global_thread >> 3U;
    if (group >= count) return;
    const unsigned int mask = 0xFFU << base;
    const unsigned int local_group = threadIdx.x >> 3U;

    std::uint8_t* state = states + static_cast<std::size_t>(group) * kStateBytes;
    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    auto& ctx = contexts[group];
    std::uint8_t* expanded = shared_keys[local_group];

    if (lane == 0U) {
        cryptonight::keccak1600(state, kStateBytes, ctx.state);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            ctx.a[i] = static_cast<std::uint8_t>(ctx.state[i] ^ ctx.state[32 + i]);
            ctx.b[i] = static_cast<std::uint8_t>(ctx.state[16 + i] ^ ctx.state[48 + i]);
        }
        ctx.tweak = cryptonight::cn_load64(state + 35) ^
                    cryptonight::cn_load64_aligned(ctx.state + 192);
        cn_expand_pseudo_key_160(ctx.state, expanded);
    }
    __syncwarp(mask);

    const std::uint8_t* text = ctx.state + 64 + static_cast<std::size_t>(lane) * 16U;
    std::uint32_t w0 = cryptonight::load32le_aligned(text);
    std::uint32_t w1 = cryptonight::load32le_aligned(text + 4);
    std::uint32_t w2 = cryptonight::load32le_aligned(text + 8);
    std::uint32_t w3 = cryptonight::load32le_aligned(text + 12);

    #pragma unroll 1
    for (std::size_t round = 0; round < init_rounds; ++round) {
        cn_aes_pseudo_words(w0, w1, w2, w3, expanded, aes_tables);
        std::uint32_t* out = reinterpret_cast<std::uint32_t*>(
            scratch + round * 128U + static_cast<std::size_t>(lane) * 16U);
        out[0] = w0; out[1] = w1; out[2] = w2; out[3] = w3;
    }
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_final_stage_word8(std::uint8_t* states,
                                              std::size_t count,
                                              std::uint8_t* scratchpads,
                                              cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    __shared__ __align__(16) std::uint8_t shared_keys[16][160];
    cryptonight::init_shared_aes_ttables(aes_tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t init_rounds = cfg.page_size / 128U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int lane = warp_lane & 7U;
    const unsigned int base = warp_lane & ~7U;
    const unsigned int group = global_thread >> 3U;
    if (group >= count) return;
    const unsigned int mask = 0xFFU << base;
    const unsigned int local_group = threadIdx.x >> 3U;

    std::uint8_t* state = states + static_cast<std::size_t>(group) * kStateBytes;
    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    auto& ctx = contexts[group];
    std::uint8_t* expanded = shared_keys[local_group];

    if (lane == 0U) cn_expand_pseudo_key_160(ctx.state + 32, expanded);
    __syncwarp(mask);

    const std::uint8_t* text = ctx.state + 64 + static_cast<std::size_t>(lane) * 16U;
    std::uint32_t w0 = cryptonight::load32le_aligned(text);
    std::uint32_t w1 = cryptonight::load32le_aligned(text + 4);
    std::uint32_t w2 = cryptonight::load32le_aligned(text + 8);
    std::uint32_t w3 = cryptonight::load32le_aligned(text + 12);

    #pragma unroll 1
    for (std::size_t round = 0; round < init_rounds; ++round) {
        const std::uint32_t* in = reinterpret_cast<const std::uint32_t*>(
            scratch + round * 128U + static_cast<std::size_t>(lane) * 16U);
        w0 ^= in[0]; w1 ^= in[1]; w2 ^= in[2]; w3 ^= in[3];
        cn_aes_pseudo_words(w0, w1, w2, w3, expanded, aes_tables);
    }

    std::uint8_t* out = ctx.state + 64 + static_cast<std::size_t>(lane) * 16U;
    cryptonight::store32le_aligned(out, w0);
    cryptonight::store32le_aligned(out + 4, w1);
    cryptonight::store32le_aligned(out + 8, w2);
    cryptonight::store32le_aligned(out + 12, w3);
    __syncwarp(mask);

    if (lane == 0U) {
        cryptonight::keccak_permute_state(ctx.state);
        std::uint8_t digest[32];
        cryptonight::dispatch_extra_hash(static_cast<std::uint8_t>(ctx.state[0] & 3U), ctx.state, digest);
        #pragma unroll
        for (int i = 0; i < 32; ++i) state[i] = digest[i];
        #pragma unroll
        for (int i = 32; i < 64; ++i) state[i] = 0;
    }
}

} // anonymous namespace
