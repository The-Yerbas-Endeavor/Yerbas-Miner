// Compile-time-specialized conventional GhostRider core kernels.
//
// The historical core512_stage kernel dispatches one of 15 algorithms through a
// runtime switch. Keeping all implementations reachable from one kernel can make
// every launch inherit the largest code/register footprint. Production uses these
// constant-algorithm kernels instead so nvcc can dead-strip the other fourteen
// hashes and tune occupancy independently per algorithm.

template <std::uint8_t Algorithm>
__global__ void core512_stage_specialized(const DeviceJob* job,
                                          const std::uint32_t* nonces,
                                          std::uint8_t* states,
                                          std::size_t count,
                                          int stage_index)
{
    static_assert(Algorithm < kCoreAlgorithmCount, "invalid GhostRider core algorithm");
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    std::uint8_t digest[64];
    bool ok = false;
    if (stage_index == 0) {
        std::uint8_t header[80];
        #pragma unroll
        for (int i = 0; i < 80; ++i) header[i] = job->header[i];
        const std::uint32_t nonce = nonces[index];
        header[76] = static_cast<std::uint8_t>(nonce);
        header[77] = static_cast<std::uint8_t>(nonce >> 8);
        header[78] = static_cast<std::uint8_t>(nonce >> 16);
        header[79] = static_cast<std::uint8_t>(nonce >> 24);
        ok = core::dispatch_core512(Algorithm, header, 80, digest);
    } else {
        ok = core::dispatch_core512(Algorithm, states + index * kStateBytes, kStateBytes, digest);
    }
    if (!ok) return;

    std::uint8_t* output = states + index * kStateBytes;
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

template <std::uint8_t Algorithm>
int core512_specialized_occupancy_threads(const cudaDeviceProp& props)
{
    int min_grid = 0;
    int block = 0;
    const cudaError_t rc = cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &block, core512_stage_specialized<Algorithm>, 0, 0);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        block = kCoreDefaultThreads;
    }
    const int warp = std::max(1, props.warpSize);
    block = std::max(warp, std::min(block, props.maxThreadsPerBlock));
    block = (block / warp) * warp;
    return std::max(warp, block);
}

int core512_specialized_threads(std::uint8_t algorithm, const cudaDeviceProp& props)
{
    switch (algorithm) {
    case 0: return core512_specialized_occupancy_threads<0>(props);
    case 1: return core512_specialized_occupancy_threads<1>(props);
    case 2: return core512_specialized_occupancy_threads<2>(props);
    case 3: return core512_specialized_occupancy_threads<3>(props);
    case 4: return core512_specialized_occupancy_threads<4>(props);
    case 5: return core512_specialized_occupancy_threads<5>(props);
    case 6: return core512_specialized_occupancy_threads<6>(props);
    case 7: return core512_specialized_occupancy_threads<7>(props);
    case 8: return core512_specialized_occupancy_threads<8>(props);
    case 9: return core512_specialized_occupancy_threads<9>(props);
    case 10: return core512_specialized_occupancy_threads<10>(props);
    case 11: return core512_specialized_occupancy_threads<11>(props);
    case 12: return core512_specialized_occupancy_threads<12>(props);
    case 13: return core512_specialized_occupancy_threads<13>(props);
    case 14: return core512_specialized_occupancy_threads<14>(props);
    default: return kCoreDefaultThreads;
    }
}

template <std::uint8_t Algorithm>
void launch_core512_specialized_one(cudaStream_t stream,
                                    const DeviceJob* job,
                                    const std::uint32_t* nonces,
                                    std::uint8_t* states,
                                    std::size_t count,
                                    int stage_index,
                                    int threads)
{
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    core512_stage_specialized<Algorithm><<<blocks, threads, 0, stream>>>(
        job, nonces, states, count, stage_index);
}

void launch_core512_specialized(cudaStream_t stream,
                                const DeviceJob* job,
                                const std::uint32_t* nonces,
                                std::uint8_t* states,
                                std::size_t count,
                                int stage_index,
                                std::uint8_t algorithm,
                                int threads)
{
    switch (algorithm) {
    case 0: launch_core512_specialized_one<0>(stream, job, nonces, states, count, stage_index, threads); break;
    case 1: launch_core512_specialized_one<1>(stream, job, nonces, states, count, stage_index, threads); break;
    case 2: launch_core512_specialized_one<2>(stream, job, nonces, states, count, stage_index, threads); break;
    case 3: launch_core512_specialized_one<3>(stream, job, nonces, states, count, stage_index, threads); break;
    case 4: launch_core512_specialized_one<4>(stream, job, nonces, states, count, stage_index, threads); break;
    case 5: launch_core512_specialized_one<5>(stream, job, nonces, states, count, stage_index, threads); break;
    case 6: launch_core512_specialized_one<6>(stream, job, nonces, states, count, stage_index, threads); break;
    case 7: launch_core512_specialized_one<7>(stream, job, nonces, states, count, stage_index, threads); break;
    case 8: launch_core512_specialized_one<8>(stream, job, nonces, states, count, stage_index, threads); break;
    case 9: launch_core512_specialized_one<9>(stream, job, nonces, states, count, stage_index, threads); break;
    case 10: launch_core512_specialized_one<10>(stream, job, nonces, states, count, stage_index, threads); break;
    case 11: launch_core512_specialized_one<11>(stream, job, nonces, states, count, stage_index, threads); break;
    case 12: launch_core512_specialized_one<12>(stream, job, nonces, states, count, stage_index, threads); break;
    case 13: launch_core512_specialized_one<13>(stream, job, nonces, states, count, stage_index, threads); break;
    case 14: launch_core512_specialized_one<14>(stream, job, nonces, states, count, stage_index, threads); break;
    default: return;
    }
}
