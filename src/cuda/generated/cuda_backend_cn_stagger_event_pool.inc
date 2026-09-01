// Persistent CUDA event pool for the staggered CryptoNight scheduler.
//
// The staggered path uses four disable-timing synchronization events on every
// production launch and two timing-capable events while benchmarking candidates.
// Creating/destroying those CUDA objects for every launch is pure host/runtime
// overhead. Keep a small fixed pool per device instead. Reuse is safe because
// each wait is enqueued after the matching record, and timing candidates fully
// synchronize their stop event before the next measurement.

namespace {

struct CnStaggerEventPool {
    std::array<cudaEvent_t, 4> sync{};
    std::array<cudaEvent_t, 2> timing{};
};

static std::array<CnStaggerEventPool, kCnPhaseProfileMaxDevices> g_cn_stagger_event_pools{};
static thread_local std::array<std::size_t, kCnPhaseProfileMaxDevices> g_cn_stagger_sync_cursor{};
static thread_local std::array<std::size_t, kCnPhaseProfileMaxDevices> g_cn_stagger_timing_cursor{};

inline cudaError_t cn_stagger_event_create_with_flags(cudaEvent_t* event, unsigned int flags)
{
    if (event == nullptr) return cudaErrorInvalidValue;

    int device_id = 0;
    const cudaError_t device_rc = cudaGetDevice(&device_id);
    if (device_rc != cudaSuccess) return device_rc;
    if (device_id < 0 || device_id >= kCnPhaseProfileMaxDevices) return cudaErrorInvalidDevice;

    const bool timing_enabled = (flags & cudaEventDisableTiming) == 0U;
    auto& pool = g_cn_stagger_event_pools[device_id];

    cudaEvent_t* slot = nullptr;
    if (timing_enabled) {
        const std::size_t index = g_cn_stagger_timing_cursor[device_id]++ % pool.timing.size();
        slot = &pool.timing[index];
    } else {
        const std::size_t index = g_cn_stagger_sync_cursor[device_id]++ % pool.sync.size();
        slot = &pool.sync[index];
    }

    if (*slot == nullptr) {
        const cudaError_t create_rc = cudaEventCreateWithFlags(slot, flags);
        if (create_rc != cudaSuccess) return create_rc;
    }

    *event = *slot;
    return cudaSuccess;
}

inline cudaError_t cn_stagger_event_release(cudaEvent_t)
{
    // Persistent by design; CUDA tears these process-lifetime handles down with
    // the context. Avoid static destructors calling CUDA during runtime teardown.
    return cudaSuccess;
}

} // anonymous namespace
