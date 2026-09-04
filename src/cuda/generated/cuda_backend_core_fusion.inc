// Experimental whole-GhostRider conventional-stage fusion.
//
// The production scan loop still schedules all 18 GhostRider stages exactly as
// before. This kernel recognizes the first conventional stage after a CryptoNight
// boundary (or stage 0), executes the entire consecutive conventional run inside
// one CUDA kernel, and keeps intermediate 512-bit digests local to the thread.
// Later scheduled conventional launches in the same run become cheap no-ops.
// CryptoNight stages remain hard synchronization/state boundaries.

namespace {

__global__ void core512_stage_fused(const DeviceJob* job,
                                    const std::uint32_t* nonces,
                                    std::uint8_t* states,
                                    std::size_t count,
                                    int stage_index,
                                    std::uint8_t algorithm)
{
    (void)algorithm;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    // If the previous stage is also conventional, that earlier production launch
    // already consumed this stage as part of its fused run.
    if (stage_index > 0) {
        const std::uint8_t previous = job->stages[stage_index - 1];
        if ((previous & ghostrider::kCryptoNightStageFlag) == 0U) return;
    }

    std::uint8_t digest_a[64];
    std::uint8_t digest_b[64];
    std::uint8_t header[80];
    const std::uint8_t* input = nullptr;
    std::size_t input_length = kStateBytes;

    if (stage_index == 0) {
        #pragma unroll
        for (int i = 0; i < 80; ++i) header[i] = job->header[i];
        const std::uint32_t nonce = nonces[index];
        header[76] = static_cast<std::uint8_t>(nonce);
        header[77] = static_cast<std::uint8_t>(nonce >> 8);
        header[78] = static_cast<std::uint8_t>(nonce >> 16);
        header[79] = static_cast<std::uint8_t>(nonce >> 24);
        input = header;
        input_length = 80U;
    } else {
        input = states + index * kStateBytes;
    }

    bool write_a = true;
    int current = stage_index;
    while (current < 18) {
        const std::uint8_t encoded = job->stages[current];
        if ((encoded & ghostrider::kCryptoNightStageFlag) != 0U) break;
        const std::uint8_t core_index = static_cast<std::uint8_t>(encoded & 0x7fU);
        std::uint8_t* output = write_a ? digest_a : digest_b;
        if (!core::dispatch_core512(core_index, input, input_length, output)) return;
        input = output;
        input_length = kStateBytes;
        write_a = !write_a;
        ++current;
    }

    std::uint8_t* state = states + index * kStateBytes;
    #pragma unroll
    for (int i = 0; i < 64; ++i) state[i] = input[i];
}

} // anonymous namespace
