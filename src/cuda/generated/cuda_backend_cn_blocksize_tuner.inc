// Runtime block-size tuner for the selected production CryptoNight phase-2 kernel.
//
// Kernel selection and exact parity remain owned by the hardened selector. This
// layer tunes launch geometry for the approved kernel and learns the production
// kernel mode from real production-sized launches. It also carries an experimental
// 8-lane cooperative T-table path; exact scratchpad parity and a measured >=5%
// production-batch win are mandatory before that path can be selected.

namespace {

constexpr int kCnBlockTuneRevision = 1;
constexpr int kCnBlockTunePasses = 3;
constexpr int kCnProductionSelectorRevision = 2;
constexpr int kCnProductionSelectorPasses = 3;
constexpr float kCnProductionPromotionRatio = 0.95F;
constexpr int kCnProductionMode8Lane = 888;
static std::array<std::array<int, 6>, kCn2LaneMaxDevices> g_cn_block_tuned_threads{};

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage_ttable8_coalesced16(
    std::size_t count,
    std::uint8_t* scratchpads,
    const cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t tables[1024];
    cryptonight::init_shared_aes_ttables(tables);
    __syncthreads();

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    constexpr std::size_t address_mask = cfg.aes_rounds - 1U;
    const unsigned int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = threadIdx.x & 7U;
    const unsigned int warp_lane = threadIdx.x & 31U;
    const unsigned int base = warp_lane & ~7U;
    const unsigned int group = global_thread >> 3U;
    if (group >= count) return;
    const unsigned int mask = 0xFFU << base;
    const unsigned int word_lane = lane >> 1U;
    const unsigned int half = lane & 1U;

    const std::uint32_t* t0 = tables;
    const std::uint32_t* t1 = tables + 256;
    const std::uint32_t* t2 = tables + 512;
    const std::uint32_t* t3 = tables + 768;

    std::uint8_t* scratch = scratchpads + static_cast<std::size_t>(group) * cfg.page_size;
    const auto& ctx = contexts[group];
    std::uint16_t aw = reinterpret_cast<const std::uint16_t*>(ctx.a)[lane];
    std::uint16_t bw = reinterpret_cast<const std::uint16_t*>(ctx.b)[lane];
    const std::uint64_t tweak = ctx.tweak;
    const std::uint32_t tweak_lo = static_cast<std::uint32_t>(tweak);
    const std::uint32_t tweak_hi = static_cast<std::uint32_t>(tweak >> 32U);

    #pragma unroll 1
    for (std::uint32_t iter = 0; iter < cfg.iterations; ++iter) {
        const auto assemble_word = [&](unsigned int pair) {
            const std::uint32_t lo = __shfl_sync(mask, static_cast<std::uint32_t>(aw), base + pair * 2U);
            const std::uint32_t hi = __shfl_sync(mask, static_cast<std::uint32_t>(aw), base + pair * 2U + 1U);
            return lo | (hi << 16U);
        };
        const std::uint32_t aw0 = assemble_word(0U);
        const std::uint32_t aw1 = assemble_word(1U);
        const std::uint32_t aw2 = assemble_word(2U);
        const std::uint32_t aw3 = assemble_word(3U);
        const std::uint64_t a_low = static_cast<std::uint64_t>(aw0) |
                                    (static_cast<std::uint64_t>(aw1) << 32U);
        const std::uint32_t index = static_cast<std::uint32_t>((a_low >> 4U) & address_mask);
        std::uint8_t* slot = scratch + static_cast<std::size_t>(index) * 16U;

        const std::uint16_t own16 = reinterpret_cast<const std::uint16_t*>(slot)[lane];
        const auto slot_word = [&](unsigned int pair) {
            const std::uint32_t lo = __shfl_sync(mask, static_cast<std::uint32_t>(own16), base + pair * 2U);
            const std::uint32_t hi = __shfl_sync(mask, static_cast<std::uint32_t>(own16), base + pair * 2U + 1U);
            return lo | (hi << 16U);
        };
        const std::uint32_t sw0 = slot_word(0U);
        const std::uint32_t sw1 = slot_word(1U);
        const std::uint32_t sw2 = slot_word(2U);
        const std::uint32_t sw3 = slot_word(3U);

        std::uint32_t cw = 0U;
        if (half == 0U) {
            const std::uint32_t words[4] = {sw0, sw1, sw2, sw3};
            const std::uint32_t av[4] = {aw0, aw1, aw2, aw3};
            const unsigned int w = word_lane;
            const std::uint32_t s0 = words[w];
            const std::uint32_t s1 = words[(w + 1U) & 3U];
            const std::uint32_t s2 = words[(w + 2U) & 3U];
            const std::uint32_t s3 = words[(w + 3U) & 3U];
            const std::uint8_t x0 = static_cast<std::uint8_t>(s0);
            const std::uint8_t x1 = static_cast<std::uint8_t>(s1 >> 8U);
            const std::uint8_t x2 = static_cast<std::uint8_t>(s2 >> 16U);
            const std::uint8_t x3 = static_cast<std::uint8_t>(s3 >> 24U);
            cw = t0[x0] ^ t1[x1] ^ t2[x2] ^ t3[x3] ^ av[w];
        }
        cw = __shfl_sync(mask, cw, base + word_lane * 2U);
        const std::uint16_t c16 = static_cast<std::uint16_t>(half == 0U ? cw : (cw >> 16U));
        reinterpret_cast<std::uint16_t*>(slot)[lane] = static_cast<std::uint16_t>(c16 ^ bw);
        __syncwarp(mask);
        if (lane == 0U) cryptonight::variant1_mutate(slot);
        __syncwarp(mask);

        const std::uint32_t c0 = __shfl_sync(mask, cw, base);
        const std::uint32_t c1 = __shfl_sync(mask, cw, base + 2U);
        const std::uint64_t c_low = static_cast<std::uint64_t>(c0) |
                                    (static_cast<std::uint64_t>(c1) << 32U);
        const std::uint32_t j = static_cast<std::uint32_t>((c_low >> 4U) & address_mask);
        std::uint8_t* slot2 = scratch + static_cast<std::size_t>(j) * 16U;

        const std::uint16_t tv16 = reinterpret_cast<const std::uint16_t*>(slot2)[lane];
        const auto tword = [&](unsigned int pair) {
            const std::uint32_t lo = __shfl_sync(mask, static_cast<std::uint32_t>(tv16), base + pair * 2U);
            const std::uint32_t hi = __shfl_sync(mask, static_cast<std::uint32_t>(tv16), base + pair * 2U + 1U);
            return lo | (hi << 16U);
        };
        const std::uint32_t tw0 = tword(0U);
        const std::uint32_t tw1 = tword(1U);
        const std::uint64_t t0v = static_cast<std::uint64_t>(tw0) |
                                  (static_cast<std::uint64_t>(tw1) << 32U);

        std::uint32_t na0_lo = 0U, na0_hi = 0U, na1_lo = 0U, na1_hi = 0U;
        if (lane == 0U) {
            const std::uint64_t lo = c_low * t0v;
            const std::uint64_t hi = __umul64hi(c_low, t0v);
            const std::uint64_t olda0 = static_cast<std::uint64_t>(aw0) |
                                        (static_cast<std::uint64_t>(aw1) << 32U);
            const std::uint64_t olda1 = static_cast<std::uint64_t>(aw2) |
                                        (static_cast<std::uint64_t>(aw3) << 32U);
            const std::uint64_t na0 = olda0 + hi;
            const std::uint64_t na1 = olda1 + lo;
            na0_lo = static_cast<std::uint32_t>(na0);
            na0_hi = static_cast<std::uint32_t>(na0 >> 32U);
            na1_lo = static_cast<std::uint32_t>(na1);
            na1_hi = static_cast<std::uint32_t>(na1 >> 32U);
        }
        na0_lo = __shfl_sync(mask, na0_lo, base);
        na0_hi = __shfl_sync(mask, na0_hi, base);
        na1_lo = __shfl_sync(mask, na1_lo, base);
        na1_hi = __shfl_sync(mask, na1_hi, base);

        const std::uint32_t store0 = na0_lo;
        const std::uint32_t store1 = na0_hi;
        const std::uint32_t store2 = na1_lo ^ tweak_lo;
        const std::uint32_t store3 = na1_hi ^ tweak_hi;
        const std::uint32_t stores[4] = {store0, store1, store2, store3};
        const std::uint32_t nas[4] = {na0_lo, na0_hi, na1_lo, na1_hi};
        const std::uint16_t store16 = static_cast<std::uint16_t>(
            half == 0U ? stores[word_lane] : (stores[word_lane] >> 16U));
        const std::uint16_t na16 = static_cast<std::uint16_t>(
            half == 0U ? nas[word_lane] : (nas[word_lane] >> 16U));
        reinterpret_cast<std::uint16_t*>(slot2)[lane] = store16;
        aw = static_cast<std::uint16_t>(na16 ^ tv16);
        bw = c16;
        __syncwarp(mask);
    }
}

template <std::uint8_t VariantIndex>
void cn_launch_production_mode(cudaStream_t stream,
                               int mode,
                               int threads,
                               std::size_t count,
                               std::uint8_t* scratchpads,
                               const cryptonight::SplitContext* contexts)
{
    if (mode != kCnProductionMode8Lane) {
        cn_launch_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
        return;
    }
    const std::size_t total_threads = count * 8U;
    const int blocks = static_cast<int>((total_threads + static_cast<std::size_t>(threads) - 1U) /
                                        static_cast<std::size_t>(threads));
    cryptonight_loop_stage_ttable8_coalesced16<VariantIndex><<<blocks, threads, 0, stream>>>(
        count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CN 8-lane loop launch failed");
}

inline const char* cn_production_mode_name(int mode)
{
    return mode == kCnProductionMode8Lane ? "8-lane-ttable16" : cn_residency_mode_name(mode);
}

template <std::uint8_t VariantIndex>
float cn_time_production_mode(cudaStream_t stream,
                              int mode,
                              int threads,
                              std::size_t count,
                              std::uint8_t* scratchpads,
                              const cryptonight::SplitContext* contexts)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate CN production test start failed");
    check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate CN production test stop failed");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord CN production test start failed");
    cn_launch_production_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord CN production test stop failed");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize CN production test failed");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime CN production test failed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

template <std::uint8_t VariantIndex>
bool cn_validate_8lane(cudaStream_t stream,
                       const std::uint8_t* scratchpads,
                       const cryptonight::SplitContext* contexts)
{
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* ref = nullptr;
    std::uint8_t* candidate = nullptr;
    cryptonight::SplitContext* ref_ctx = nullptr;
    cryptonight::SplitContext* candidate_ctx = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ref), cfg.page_size), "cudaMalloc CN 8-lane parity ref failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate), cfg.page_size), "cudaMalloc CN 8-lane parity candidate failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ref_ctx), sizeof(cryptonight::SplitContext)), "cudaMalloc CN 8-lane parity ref ctx failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&candidate_ctx), sizeof(cryptonight::SplitContext)), "cudaMalloc CN 8-lane parity candidate ctx failed");
    check_cuda(cudaMemcpyAsync(ref, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 8-lane parity ref failed");
    check_cuda(cudaMemcpyAsync(candidate, scratchpads, cfg.page_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 8-lane parity candidate failed");
    check_cuda(cudaMemcpyAsync(ref_ctx, contexts, sizeof(cryptonight::SplitContext), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 8-lane parity ref ctx failed");
    check_cuda(cudaMemcpyAsync(candidate_ctx, contexts, sizeof(cryptonight::SplitContext), cudaMemcpyDeviceToDevice, stream), "cudaMemcpy CN 8-lane parity candidate ctx failed");
    cn_launch_mode<VariantIndex>(stream, 44, 32, 1U, ref, ref_ctx);
    cn_launch_production_mode<VariantIndex>(stream, kCnProductionMode8Lane, 32, 1U, candidate, candidate_ctx);
    std::vector<std::uint8_t> host_ref(cfg.page_size), host_candidate(cfg.page_size);
    check_cuda(cudaMemcpyAsync(host_ref.data(), ref, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy CN 8-lane parity ref result failed");
    check_cuda(cudaMemcpyAsync(host_candidate.data(), candidate, cfg.page_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpy CN 8-lane parity candidate result failed");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize CN 8-lane parity failed");
    cudaFree(candidate_ctx);
    cudaFree(ref_ctx);
    cudaFree(candidate);
    cudaFree(ref);
    return host_ref == host_candidate;
}

struct CnProductionSelectorState {
    bool initialized{false};
    bool selected{false};
    bool tile_ok{false};
    bool cg_ok{false};
    bool wide_ok{false};
    std::size_t count{0};
    int threads_ref{128};
    int threads_t4{128};
    int threads_tile{128};
    int threads_cg{128};
    int threads_wide{128};
    int baseline_samples{0};
    int tile_samples{0};
    int cg_samples{0};
    int wide_samples{0};
    std::array<float, kCnProductionSelectorPasses> baseline_ms{};
    std::array<float, kCnProductionSelectorPasses> tile_ms{};
    std::array<float, kCnProductionSelectorPasses> cg_ms{};
    std::array<float, kCnProductionSelectorPasses> wide_ms{};
};

static std::array<std::array<CnProductionSelectorState, 6>, kCn2LaneMaxDevices>
    g_cn_production_selector{};

template <std::uint8_t VariantIndex>
std::filesystem::path cn_block_tune_cache_path(int device_id,
                                               const cudaDeviceProp& props,
                                               std::size_t batch_size,
                                               int mode)
{
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, batch_size, driver_version, runtime_version);
    return cn_cache_directory() /
        ("cn-block-rev" + std::to_string(kCnBlockTuneRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) +
         "-m" + std::to_string(mode) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_block_tune_cache(int device_id,
                              const cudaDeviceProp& props,
                              std::size_t batch_size,
                              int mode)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_block_tune_cache_path<VariantIndex>(device_id, props, batch_size, mode);
    if (path.empty()) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0, variant = -1, cached_mode = 0, threads = 0;
    if (!(in >> magic >> revision >> variant >> cached_mode >> threads) ||
        magic != "YERBAS_CN_BLOCK" || revision != kCnBlockTuneRevision ||
        variant != static_cast<int>(VariantIndex) || cached_mode != mode ||
        threads < props.warpSize || threads > props.maxThreadsPerBlock ||
        (threads % props.warpSize) != 0)
        return false;
    g_cn_block_tuned_threads[device_id][VariantIndex] = threads;
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_block_tune_cache(int device_id,
                              const cudaDeviceProp& props,
                              std::size_t batch_size,
                              int mode,
                              int threads)
{
    try {
        const auto path = cn_block_tune_cache_path<VariantIndex>(device_id, props, batch_size, mode);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (out)
            out << "YERBAS_CN_BLOCK " << kCnBlockTuneRevision << ' '
                << static_cast<int>(VariantIndex) << ' ' << mode << ' ' << threads << '\n';
    } catch (...) {}
}

template <std::uint8_t VariantIndex>
std::filesystem::path cn_production_selector_cache_path(int device_id,
                                                        const cudaDeviceProp& props,
                                                        std::size_t batch_size)
{
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, batch_size, driver_version, runtime_version);
    return cn_cache_directory() /
        ("cn-production-rev" + std::to_string(kCnProductionSelectorRevision) +
         "-v" + std::to_string(static_cast<unsigned int>(VariantIndex)) + '-' + key + ".txt");
}

template <std::uint8_t VariantIndex>
bool load_cn_production_selector_cache(int device_id,
                                       const cudaDeviceProp& props,
                                       std::size_t count)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;
    const auto path = cn_production_selector_cache_path<VariantIndex>(device_id, props, count);
    if (path.empty()) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0, variant = -1, mode = 0, threads = 0;
    const int warp = std::max(32, props.warpSize);
    if (!(in >> magic >> revision >> variant >> mode >> threads) ||
        magic != "YERBAS_CN_PRODUCTION" || revision != kCnProductionSelectorRevision ||
        variant != static_cast<int>(VariantIndex) ||
        (mode != 4 && mode != 44 && mode != 222 && mode != 444 && mode != kCnProductionMode8Lane) ||
        threads < warp || threads > props.maxThreadsPerBlock || (threads % warp) != 0)
        return false;
    if (mode == 222) {
        int active = 0;
        const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, cryptonight_loop_stage_ttable2_tile64<VariantIndex>, threads,
            cn_tile64_dynamic_shared_bytes(threads));
        if (rc != cudaSuccess || active <= 0) { cudaGetLastError(); return false; }
    }
    auto& state = g_cn_production_selector[device_id][VariantIndex];
    state.initialized = true;
    state.selected = true;
    state.count = count;
    g_cn_hardened_mode[device_id][VariantIndex] = mode;
    g_cn_hardened_threads[device_id][VariantIndex] = threads;
    std::cout << "[GPU " << device_id << "] CryptoNight production selector cache loaded | "
              << cryptonight::config_value(VariantIndex).name
              << " | batch=" << count
              << " | production=" << cn_production_mode_name(mode)
              << " | threads=" << threads << '\n';
    return true;
}

template <std::uint8_t VariantIndex>
void save_cn_production_selector_cache(int device_id,
                                       const cudaDeviceProp& props,
                                       std::size_t count,
                                       int mode,
                                       int threads)
{
    try {
        const auto path = cn_production_selector_cache_path<VariantIndex>(device_id, props, count);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_PRODUCTION " << kCnProductionSelectorRevision << ' '
            << static_cast<int>(VariantIndex) << ' ' << mode << ' ' << threads << '\n';
        out.close();
        if (!out) return;
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (...) {}
}

template <std::uint8_t VariantIndex>
void initialize_cn_production_selector(cudaStream_t stream,
                                       int device_id,
                                       const cudaDeviceProp& props,
                                       std::size_t count,
                                       const std::uint8_t* scratchpads,
                                       const cryptonight::SplitContext* contexts)
{
    auto& state = g_cn_production_selector[device_id][VariantIndex];
    if (state.initialized) return;
    if (load_cn_production_selector_cache<VariantIndex>(device_id, props, count)) return;
    state = CnProductionSelectorState{};
    state.count = count;
    const int warp = std::max(32, props.warpSize);
    state.threads_ref = cn_occupancy_block_size(
        cryptonight_loop_stage_cooperative<VariantIndex>, warp, props.maxThreadsPerBlock);
    state.threads_t4 = cn_occupancy_block_size(
        cryptonight_loop_stage_ttable4_coalesced<VariantIndex>, warp, props.maxThreadsPerBlock);
    state.threads_cg = cn_occupancy_block_size(
        cryptonight_loop_stage_ttable4_cg<VariantIndex>, warp, props.maxThreadsPerBlock);
    state.threads_tile = cn_tile64_block_size<VariantIndex>(props);
    state.threads_wide = cn_occupancy_block_size(
        cryptonight_loop_stage_ttable8_coalesced16<VariantIndex>, warp, props.maxThreadsPerBlock);

    const bool parity_t4 = cn_validate_pair<VariantIndex>(
        stream, 4, 32, 44, 32, scratchpads, contexts);
    if (!parity_t4) {
        state.initialized = true;
        state.selected = true;
        g_cn_hardened_mode[device_id][VariantIndex] = 4;
        g_cn_hardened_threads[device_id][VariantIndex] = state.threads_ref;
        save_cn_production_selector_cache<VariantIndex>(device_id, props, count, 4, state.threads_ref);
        std::cout << "[CUDA CN production selector] GPU " << device_id
                  << " | " << cryptonight::config_value(VariantIndex).name
                  << " | batch=" << count
                  << " | baseline-parity=FAIL | production=4-lane-ref\n";
        return;
    }
    state.tile_ok = cn_validate_pair<VariantIndex>(stream, 44, 32, 222, 32, scratchpads, contexts);
    state.cg_ok = cn_validate_pair<VariantIndex>(stream, 44, 32, 444, 32, scratchpads, contexts);
    state.wide_ok = cn_validate_8lane<VariantIndex>(stream, scratchpads, contexts);
    state.initialized = true;

    std::cout << "[CUDA CN production selector] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | batch=" << count
              << " | passes=" << kCnProductionSelectorPasses
              << " | threshold=5%"
              << " | 8lane-parity=" << (state.wide_ok ? "PASS" : "FAIL")
              << " | tile64-parity=" << (state.tile_ok ? "PASS" : "FAIL")
              << " | cg-parity=" << (state.cg_ok ? "PASS" : "FAIL")
              << " | sampling=real-production-batch\n";
}

inline int cn_production_trial_mode(const CnProductionSelectorState& state)
{
    if (state.baseline_samples < kCnProductionSelectorPasses) return 44;
    if (state.wide_ok && state.wide_samples < kCnProductionSelectorPasses) return kCnProductionMode8Lane;
    if (state.tile_ok && state.tile_samples < kCnProductionSelectorPasses) return 222;
    if (state.cg_ok && state.cg_samples < kCnProductionSelectorPasses) return 444;
    return 44;
}

inline int cn_production_trial_threads(const CnProductionSelectorState& state, int mode)
{
    if (mode == kCnProductionMode8Lane) return state.threads_wide;
    if (mode == 222) return state.threads_tile;
    if (mode == 444) return state.threads_cg;
    if (mode == 44) return state.threads_t4;
    return state.threads_ref;
}

template <std::uint8_t VariantIndex>
void record_cn_production_sample(int device_id,
                                 const cudaDeviceProp& props,
                                 int mode,
                                 float elapsed_ms)
{
    auto& state = g_cn_production_selector[device_id][VariantIndex];
    if (mode == 44 && state.baseline_samples < kCnProductionSelectorPasses)
        state.baseline_ms[state.baseline_samples++] = elapsed_ms;
    else if (mode == kCnProductionMode8Lane && state.wide_samples < kCnProductionSelectorPasses)
        state.wide_ms[state.wide_samples++] = elapsed_ms;
    else if (mode == 222 && state.tile_samples < kCnProductionSelectorPasses)
        state.tile_ms[state.tile_samples++] = elapsed_ms;
    else if (mode == 444 && state.cg_samples < kCnProductionSelectorPasses)
        state.cg_ms[state.cg_samples++] = elapsed_ms;

    const bool done = state.baseline_samples >= kCnProductionSelectorPasses &&
        (!state.wide_ok || state.wide_samples >= kCnProductionSelectorPasses) &&
        (!state.tile_ok || state.tile_samples >= kCnProductionSelectorPasses) &&
        (!state.cg_ok || state.cg_samples >= kCnProductionSelectorPasses);
    if (!done) return;

    const float baseline = cn_hardened_median(state.baseline_ms);
    const float wide = state.wide_ok ? cn_hardened_median(state.wide_ms) : -1.0F;
    const float tile = state.tile_ok ? cn_hardened_median(state.tile_ms) : -1.0F;
    const float cg = state.cg_ok ? cn_hardened_median(state.cg_ms) : -1.0F;
    int best_mode = 44;
    int best_threads = state.threads_t4;
    float best_ms = baseline;
    if (state.wide_ok && wide < baseline * kCnProductionPromotionRatio && wide < best_ms) {
        best_mode = kCnProductionMode8Lane;
        best_threads = state.threads_wide;
        best_ms = wide;
    }
    if (state.tile_ok && tile < baseline * kCnProductionPromotionRatio && tile < best_ms) {
        best_mode = 222;
        best_threads = state.threads_tile;
        best_ms = tile;
    }
    if (state.cg_ok && cg < baseline * kCnProductionPromotionRatio && cg < best_ms) {
        best_mode = 444;
        best_threads = state.threads_cg;
        best_ms = cg;
    }

    state.selected = true;
    g_cn_hardened_mode[device_id][VariantIndex] = best_mode;
    g_cn_hardened_threads[device_id][VariantIndex] = best_threads;
    save_cn_production_selector_cache<VariantIndex>(device_id, props, state.count, best_mode, best_threads);

    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN production selector] GPU " << device_id
              << " | " << cryptonight::config_value(VariantIndex).name
              << " | batch=" << state.count
              << " | baseline=" << baseline << " ms@" << state.threads_t4
              << " | 8lane=" << wide << " ms@" << state.threads_wide
              << " | tile64=" << tile << " ms@" << state.threads_tile
              << " | cg=" << cg << " ms@" << state.threads_cg
              << " | production=" << cn_production_mode_name(best_mode)
              << " | selected=" << best_ms << " ms"
              << std::defaultfloat << '\n';
}

template <std::uint8_t VariantIndex>
int cn_tune_selected_block_size(cudaStream_t stream,
                                int device_id,
                                const cudaDeviceProp& props,
                                int mode,
                                int baseline_threads,
                                std::size_t count,
                                std::uint8_t* scratchpads,
                                const cryptonight::SplitContext* contexts)
{
    if (g_cn_block_tuned_threads[device_id][VariantIndex] > 0)
        return g_cn_block_tuned_threads[device_id][VariantIndex];
    if (load_cn_block_tune_cache<VariantIndex>(device_id, props, count, mode))
        return g_cn_block_tuned_threads[device_id][VariantIndex];

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::size_t free_bytes = 0, total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo CN block tuner failed");
    (void)total_bytes;
    const std::size_t by_memory = cfg.page_size ? (free_bytes / 20U) / cfg.page_size : 0U;
    const std::size_t sample_count = std::max<std::size_t>(
        1U, std::min<std::size_t>(count, std::min<std::size_t>(32U, by_memory)));

    std::uint8_t* clone = nullptr;
    cryptonight::SplitContext* clone_ctx = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone), sample_count * cfg.page_size),
               "cudaMalloc CN block tuner scratch failed");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&clone_ctx),
                          sample_count * sizeof(cryptonight::SplitContext)),
               "cudaMalloc CN block tuner context failed");
    auto reset = [&]() {
        check_cuda(cudaMemcpyAsync(clone, scratchpads, sample_count * cfg.page_size,
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN block tuner scratch failed");
        check_cuda(cudaMemcpyAsync(clone_ctx, contexts,
                                   sample_count * sizeof(cryptonight::SplitContext),
                                   cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpy CN block tuner context failed");
    };

    std::vector<int> candidates;
    const auto add = [&](int threads) {
        if (threads < props.warpSize || threads > props.maxThreadsPerBlock ||
            (threads % props.warpSize) != 0) return;
        if (mode == 222) {
            int active = 0;
            const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, cryptonight_loop_stage_ttable2_tile64<VariantIndex>, threads,
                cn_tile64_dynamic_shared_bytes(threads));
            if (rc != cudaSuccess || active <= 0) { cudaGetLastError(); return; }
        }
        if (std::find(candidates.begin(), candidates.end(), threads) == candidates.end())
            candidates.push_back(threads);
    };
    add(baseline_threads);
    for (int threads = props.warpSize; threads <= std::min(256, props.maxThreadsPerBlock);
         threads += props.warpSize)
        add(threads);

    int best_threads = baseline_threads;
    float best_ms = 1.0e30F;
    for (const int threads : candidates) {
        reset();
        (void)cn_time_production_mode<VariantIndex>(stream, mode, threads, sample_count, clone, clone_ctx);
        std::array<float, kCnBlockTunePasses> times{};
        for (int pass = 0; pass < kCnBlockTunePasses; ++pass) {
            reset();
            times[pass] = cn_time_production_mode<VariantIndex>(stream, mode, threads, sample_count, clone, clone_ctx);
        }
        const float median = cn_hardened_median(times);
        if (median < best_ms) {
            best_ms = median;
            best_threads = threads;
        }
    }

    cudaFree(clone_ctx);
    cudaFree(clone);
    g_cn_block_tuned_threads[device_id][VariantIndex] = best_threads;
    save_cn_block_tune_cache<VariantIndex>(device_id, props, count, mode, best_threads);
    std::cout << std::fixed << std::setprecision(3)
              << "[CUDA CN block tuner] GPU " << device_id
              << " | " << cfg.name
              << " | mode=" << cn_production_mode_name(mode)
              << " | baseline=" << baseline_threads
              << " | selected=" << best_threads
              << " | median=" << best_ms << " ms"
              << " | candidates=" << candidates.size()
              << std::defaultfloat << '\n';
    return best_threads;
}

template <std::uint8_t VariantIndex, bool UseTTable>
void launch_cn_loop_block_tuned(cudaStream_t stream,
                                std::uint8_t* states,
                                std::size_t count,
                                std::uint8_t* scratchpads,
                                cryptonight::SplitContext* contexts,
                                const CnGeometry& geometry)
{
    (void)states;
    (void)geometry;
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice CN block tuner failed");
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices) {
        cn_launch_mode<VariantIndex>(stream, 4, 128, count, scratchpads, contexts);
        return;
    }

    cudaDeviceProp props{};
    check_cuda(cudaGetDeviceProperties(&props, device_id), "cudaGetDeviceProperties CN block tuner failed");
    initialize_cn_production_selector<VariantIndex>(stream, device_id, props, count, scratchpads, contexts);
    auto& selector = g_cn_production_selector[device_id][VariantIndex];

    if (!selector.selected && selector.count != count) {
        cn_launch_mode<VariantIndex>(stream, 44, selector.threads_t4, count, scratchpads, contexts);
        return;
    }

    if (!selector.selected) {
        const int mode = cn_production_trial_mode(selector);
        const int threads = cn_production_trial_threads(selector, mode);
        cudaEvent_t start{}, stop{};
        check_cuda(cudaEventCreateWithFlags(&start, cudaEventDefault), "cudaEventCreate production selector start failed");
        check_cuda(cudaEventCreateWithFlags(&stop, cudaEventDefault), "cudaEventCreate production selector stop failed");
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord production selector start failed");
        cn_launch_production_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord production selector stop failed");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize production selector failed");
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime production selector failed");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        record_cn_production_sample<VariantIndex>(device_id, props, mode, elapsed_ms);
        return;
    }

    int mode = g_cn_hardened_mode[device_id][VariantIndex];
    int baseline_threads = g_cn_hardened_threads[device_id][VariantIndex];
    if (mode == 0) mode = 4;
    if (baseline_threads <= 0) baseline_threads = 128;
    const int threads = cn_tune_selected_block_size<VariantIndex>(
        stream, device_id, props, mode, baseline_threads, count, scratchpads, contexts);
    cn_launch_production_mode<VariantIndex>(stream, mode, threads, count, scratchpads, contexts);
}

} // anonymous namespace
