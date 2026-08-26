// Generic CN-Fast cooperative v9 latency-hiding candidate.
//
// One four-lane cooperative group advances two independent hashes.  The two
// dependency chains remain semantically independent, but memory reads and
// arithmetic from hash B are exposed while hash A is waiting on scratchpad
// latency (and vice versa).  No GPU model assumptions are made; runtime parity
// and performance qualification decide whether this kernel is used.

namespace {

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_coop4_v9_pair(std::size_t count,
                                                      std::uint8_t* scratchpads,
                                                      const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;

    const unsigned int gt = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 3U;
    const unsigned int wl = threadIdx.x & 31U;
    const unsigned int base = wl & ~3U;
    const unsigned int pair = gt >> 2;
    const std::size_t hash0 = static_cast<std::size_t>(pair) * 2U;
    if (hash0 >= count) return;
    const std::size_t hash1 = hash0 + 1U;
    const bool have1 = hash1 < count;
    const unsigned int mask = 0xFU << base;

    std::uint8_t* scratch0 = scratchpads + hash0 * cfg.page_size;
    std::uint8_t* scratch1 = have1 ? scratchpads + hash1 * cfg.page_size : scratch0;
    const auto& ctx0 = contexts[hash0];
    const auto& ctx1 = have1 ? contexts[hash1] : contexts[hash0];

    std::uint32_t aw0 = cryptonight::load32le_aligned(ctx0.a + lane * 4U);
    std::uint32_t bw0 = cryptonight::load32le_aligned(ctx0.b + lane * 4U);
    std::uint32_t aw1 = have1 ? cryptonight::load32le_aligned(ctx1.a + lane * 4U) : aw0;
    std::uint32_t bw1 = have1 ? cryptonight::load32le_aligned(ctx1.b + lane * 4U) : bw0;
    const std::uint64_t tweak0 = ctx0.tweak;
    const std::uint64_t tweak1 = have1 ? ctx1.tweak : tweak0;

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        // Resolve both first dependent addresses before consuming either load.
        const std::uint32_t a00 = __shfl_sync(mask, aw0, base);
        const std::uint32_t a01 = __shfl_sync(mask, aw0, base + 1U);
        const std::uint32_t a10 = __shfl_sync(mask, aw1, base);
        const std::uint32_t a11 = __shfl_sync(mask, aw1, base + 1U);
        const std::uint64_t a0_low = static_cast<std::uint64_t>(a00) |
                                     (static_cast<std::uint64_t>(a01) << 32U);
        const std::uint64_t a1_low = static_cast<std::uint64_t>(a10) |
                                     (static_cast<std::uint64_t>(a11) << 32U);
        const std::uint32_t index0 = static_cast<std::uint32_t>((a0_low >> 4U) & address_mask);
        const std::uint32_t index1 = static_cast<std::uint32_t>((a1_low >> 4U) & address_mask);
        std::uint8_t* slot0 = scratch0 + static_cast<std::size_t>(index0) * 16U;
        std::uint8_t* slot1 = scratch1 + static_cast<std::size_t>(index1) * 16U;

        // Issue the two independent scratchpad reads back-to-back.  Each lane
        // owns one AES column for each hash.
        std::uint8_t x00, x01, x02, x03;
        std::uint8_t x10, x11, x12, x13;
        if (lane == 0U) {
            x00=slot0[0];  x01=slot0[5];  x02=slot0[10]; x03=slot0[15];
            x10=slot1[0];  x11=slot1[5];  x12=slot1[10]; x13=slot1[15];
        } else if (lane == 1U) {
            x00=slot0[4];  x01=slot0[9];  x02=slot0[14]; x03=slot0[3];
            x10=slot1[4];  x11=slot1[9];  x12=slot1[14]; x13=slot1[3];
        } else if (lane == 2U) {
            x00=slot0[8];  x01=slot0[13]; x02=slot0[2];  x03=slot0[7];
            x10=slot1[8];  x11=slot1[13]; x12=slot1[2];  x13=slot1[7];
        } else {
            x00=slot0[12]; x01=slot0[1];  x02=slot0[6];  x03=slot0[11];
            x10=slot1[12]; x11=slot1[1];  x12=slot1[6];  x13=slot1[11];
        }

        const std::uint32_t cw0 = coop_mix_column(
            cryptonight::aes_sbox(x00), cryptonight::aes_sbox(x01),
            cryptonight::aes_sbox(x02), cryptonight::aes_sbox(x03)) ^ aw0;
        const std::uint32_t cw1 = coop_mix_column(
            cryptonight::aes_sbox(x10), cryptonight::aes_sbox(x11),
            cryptonight::aes_sbox(x12), cryptonight::aes_sbox(x13)) ^ aw1;

        reinterpret_cast<std::uint32_t*>(slot0)[lane] = cw0 ^ bw0;
        if (have1) reinterpret_cast<std::uint32_t*>(slot1)[lane] = cw1 ^ bw1;
        __syncwarp(mask);
        if (lane == 0U) {
            cryptonight::variant1_mutate(slot0);
            if (have1) cryptonight::variant1_mutate(slot1);
        }
        __syncwarp(mask);

        const std::uint32_t c00 = __shfl_sync(mask, cw0, base);
        const std::uint32_t c01 = __shfl_sync(mask, cw0, base + 1U);
        const std::uint32_t c10 = __shfl_sync(mask, cw1, base);
        const std::uint32_t c11 = __shfl_sync(mask, cw1, base + 1U);

        const std::uint32_t old00 = a00;
        const std::uint32_t old01 = a01;
        const std::uint32_t old02 = __shfl_sync(mask, aw0, base + 2U);
        const std::uint32_t old03 = __shfl_sync(mask, aw0, base + 3U);
        const std::uint32_t old10 = a10;
        const std::uint32_t old11 = a11;
        const std::uint32_t old12 = __shfl_sync(mask, aw1, base + 2U);
        const std::uint32_t old13 = __shfl_sync(mask, aw1, base + 3U);

        std::uint32_t next00=0U,next01=0U,next02=0U,next03=0U;
        std::uint32_t next10=0U,next11=0U,next12=0U,next13=0U;
        if (lane == 0U) {
            const std::uint64_t c0_low = static_cast<std::uint64_t>(c00) |
                                         (static_cast<std::uint64_t>(c01) << 32U);
            const std::uint64_t c1_low = static_cast<std::uint64_t>(c10) |
                                         (static_cast<std::uint64_t>(c11) << 32U);
            const std::uint32_t j0 = static_cast<std::uint32_t>((c0_low >> 4U) & address_mask);
            const std::uint32_t j1 = static_cast<std::uint32_t>((c1_low >> 4U) & address_mask);
            std::uint8_t* slot20 = scratch0 + static_cast<std::size_t>(j0) * 16U;
            std::uint8_t* slot21 = scratch1 + static_cast<std::size_t>(j1) * 16U;

            // Load both second dependent slots before beginning either multiply.
            const std::uint64_t t00 = cryptonight::cn_load64_aligned(slot20);
            const std::uint64_t t01 = cryptonight::cn_load64_aligned(slot20 + 8);
            const std::uint64_t t10 = have1 ? cryptonight::cn_load64_aligned(slot21) : 0U;
            const std::uint64_t t11 = have1 ? cryptonight::cn_load64_aligned(slot21 + 8) : 0U;

            const std::uint64_t lo0 = c0_low * t00;
            const std::uint64_t hi0 = __umul64hi(c0_low, t00);
            const std::uint64_t lo1 = c1_low * t10;
            const std::uint64_t hi1 = __umul64hi(c1_low, t10);

            const std::uint64_t olda00 = static_cast<std::uint64_t>(old00) |
                                          (static_cast<std::uint64_t>(old01) << 32U);
            const std::uint64_t olda01 = static_cast<std::uint64_t>(old02) |
                                          (static_cast<std::uint64_t>(old03) << 32U);
            const std::uint64_t olda10 = static_cast<std::uint64_t>(old10) |
                                          (static_cast<std::uint64_t>(old11) << 32U);
            const std::uint64_t olda11 = static_cast<std::uint64_t>(old12) |
                                          (static_cast<std::uint64_t>(old13) << 32U);

            const std::uint64_t na00 = olda00 + hi0;
            const std::uint64_t na01 = olda01 + lo0;
            cryptonight::cn_store64_aligned(slot20, na00);
            cryptonight::cn_store64_aligned(slot20 + 8, na01 ^ tweak0);
            const std::uint64_t xa00 = na00 ^ t00;
            const std::uint64_t xa01 = na01 ^ t01;
            next00=static_cast<std::uint32_t>(xa00);
            next01=static_cast<std::uint32_t>(xa00 >> 32U);
            next02=static_cast<std::uint32_t>(xa01);
            next03=static_cast<std::uint32_t>(xa01 >> 32U);

            if (have1) {
                const std::uint64_t na10 = olda10 + hi1;
                const std::uint64_t na11 = olda11 + lo1;
                cryptonight::cn_store64_aligned(slot21, na10);
                cryptonight::cn_store64_aligned(slot21 + 8, na11 ^ tweak1);
                const std::uint64_t xa10 = na10 ^ t10;
                const std::uint64_t xa11 = na11 ^ t11;
                next10=static_cast<std::uint32_t>(xa10);
                next11=static_cast<std::uint32_t>(xa10 >> 32U);
                next12=static_cast<std::uint32_t>(xa11);
                next13=static_cast<std::uint32_t>(xa11 >> 32U);
            }
        }

        const std::uint32_t out00=__shfl_sync(mask,next00,base);
        const std::uint32_t out01=__shfl_sync(mask,next01,base);
        const std::uint32_t out02=__shfl_sync(mask,next02,base);
        const std::uint32_t out03=__shfl_sync(mask,next03,base);
        aw0 = lane==0U?out00:lane==1U?out01:lane==2U?out02:out03;
        bw0 = cw0;

        if (have1) {
            const std::uint32_t out10=__shfl_sync(mask,next10,base);
            const std::uint32_t out11=__shfl_sync(mask,next11,base);
            const std::uint32_t out12=__shfl_sync(mask,next12,base);
            const std::uint32_t out13=__shfl_sync(mask,next13,base);
            aw1 = lane==0U?out10:lane==1U?out11:lane==2U?out12:out13;
            bw1 = cw1;
        }
        __syncwarp(mask);
    }
}

} // anonymous namespace
