// Shared-key production CryptoNight setup/final kernels.
//
// The cooperative 8-lane setup/final path has eight independent 16-byte AES
// chains per hash. The previous implementation expanded the same AES-256 key
// independently in every lane, creating eight copies of a 240-byte local key
// schedule per hash. Besides duplicated arithmetic, that large per-thread local
// array can increase register/local-memory pressure.
//
// These kernels expand each AES key once per hash in lane 0 and publish the
// 240-byte schedule through shared memory to the other seven lanes. Each
// 8-lane subgroup owns a separate shared schedule, so hashes remain independent
// and the cryptographic byte order is unchanged.

namespace {

template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_setup_stage_cooperative8_sharedkey(std::uint8_t* states,
                                                               std::size_t count,
                                                               std::uint8_t* scratchpads,
                                                               cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    __shared__ __align__(16) std::uint8_t shared_keys[16][240];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }

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

    // Per-hash initialization and AES key expansion are performed exactly once.
    // The subgroup synchronization publishes both the SplitContext fields and
    // the shared key schedule to the seven cooperating AES-chain lanes.
    if (lane == 0U) {
        cryptonight::keccak1600(state, kStateBytes, ctx.state);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            ctx.a[i] = static_cast<std::uint8_t>(ctx.state[i] ^ ctx.state[32 + i]);
            ctx.b[i] = static_cast<std::uint8_t>(ctx.state[16 + i] ^ ctx.state[48 + i]);
        }
        ctx.tweak = cryptonight::cn_load64(state + 35) ^
                    cryptonight::cn_load64_aligned(ctx.state + 192);
        cryptonight::aes256_expand_key(ctx.state, expanded);
    }
    __syncwarp(mask);

    alignas(16) std::uint8_t text[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        text[i] = ctx.state[64 + static_cast<int>(lane) * 16 + i];

    #pragma unroll 1
    for (std::size_t round = 0; round < init_rounds; ++round) {
        if constexpr (UseTTable)
            cryptonight::aes_pseudo_round_ttable(text, expanded, aes_tables);
        else
            cryptonight::aes_pseudo_round(text, expanded);

        std::uint8_t* out = scratch + round * 128U + static_cast<std::size_t>(lane) * 16U;
        #pragma unroll
        for (int i = 0; i < 16; ++i) out[i] = text[i];
    }
}

template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_final_stage_cooperative8_sharedkey(std::uint8_t* states,
                                                               std::size_t count,
                                                               std::uint8_t* scratchpads,
                                                               cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    __shared__ __align__(16) std::uint8_t shared_keys[16][240];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }

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

    if (lane == 0U)
        cryptonight::aes256_expand_key(ctx.state + 32, expanded);
    __syncwarp(mask);

    alignas(16) std::uint8_t text[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        text[i] = ctx.state[64 + static_cast<int>(lane) * 16 + i];

    #pragma unroll 1
    for (std::size_t round = 0; round < init_rounds; ++round) {
        const std::uint8_t* in = scratch + round * 128U + static_cast<std::size_t>(lane) * 16U;
        #pragma unroll
        for (int i = 0; i < 16; ++i) text[i] ^= in[i];

        if constexpr (UseTTable)
            cryptonight::aes_pseudo_round_ttable(text, expanded, aes_tables);
        else
            cryptonight::aes_pseudo_round(text, expanded);
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
        ctx.state[64 + static_cast<int>(lane) * 16 + i] = text[i];
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
