// Batch-aware runtime state for CryptoNight production tuning.
//
// Rotation-aware batching means the same CN variant can execute at several
// production counts depending on its two companion variants. The disk caches
// already include batch size in their keys, but the live selector state is held
// in one slot per GPU/variant. Preserve that live state per batch so switching
// rotations neither reuses a selection measured at another count nor throws away
// an in-progress retune. Eight slots are sufficient because the rotation batch is
// always one of the six per-variant preferred counts (or a conservative class
// fallback), while leaving headroom for transition/fallback values.

namespace {

constexpr std::size_t kGpuVariantRuntimeBatchSlots = 8;

struct GpuVariantRuntimeBatchSlot {
    bool used{false};
    std::size_t count{0};
    CnProductionSelectorState selector{};
    CnProductionGeometryState geometry{};
    int residency_mode{0};
    int residency_threads{0};
    int ttable4_parity{0};
    int tile64_parity{0};
    int cg_parity{0};
    int hardened_mode{0};
    int hardened_threads{0};
    int block_threads{0};
    int setup_mode{0};
    int final_mode{0};
    int overlap_mode{0};
    int stagger_mode{0};
    bool overlap_reported{false};
    bool stagger_reported{false};
};

static std::array<
    std::array<
        std::array<GpuVariantRuntimeBatchSlot, kGpuVariantRuntimeBatchSlots>,
        kGpuVariantBatchCount>,
    kCn2LaneMaxDevices> g_gpu_variant_runtime_slots{};

static std::array<std::array<std::size_t, kGpuVariantBatchCount>,
                  kCn2LaneMaxDevices> g_gpu_variant_runtime_active_count{};

GpuVariantRuntimeBatchSlot* gpu_variant_runtime_find_slot(int device_id,
                                                          std::size_t variant,
                                                          std::size_t count)
{
    auto& slots = g_gpu_variant_runtime_slots[device_id][variant];
    for (auto& slot : slots)
        if (slot.used && slot.count == count) return &slot;
    return nullptr;
}

GpuVariantRuntimeBatchSlot* gpu_variant_runtime_writable_slot(int device_id,
                                                              std::size_t variant,
                                                              std::size_t count)
{
    if (auto* existing = gpu_variant_runtime_find_slot(device_id, variant, count))
        return existing;

    auto& slots = g_gpu_variant_runtime_slots[device_id][variant];
    for (auto& slot : slots) {
        if (!slot.used) {
            slot.used = true;
            slot.count = count;
            return &slot;
        }
    }

    // Production can only expose a small finite set of rotation batches. This is
    // a defensive fallback for unexpected future policies: replace a deterministic
    // slot rather than failing mining. Exact disk caches remain the source of truth
    // after process restart.
    auto& slot = slots[count % slots.size()];
    slot = GpuVariantRuntimeBatchSlot{};
    slot.used = true;
    slot.count = count;
    return &slot;
}

void gpu_variant_runtime_save_active(int device_id,
                                     std::size_t variant,
                                     std::size_t count)
{
    if (count == 0U) return;
    auto* slot = gpu_variant_runtime_writable_slot(device_id, variant, count);
    slot->selector = g_cn_production_selector[device_id][variant];
    slot->geometry = g_cn_production_geometry[device_id][variant];
    slot->residency_mode = g_cn_residency_mode[device_id][variant];
    slot->residency_threads = g_cn_residency_threads[device_id][variant];
    slot->ttable4_parity = g_cn_ttable4_parity[device_id][variant];
    slot->tile64_parity = g_cn_tile64_parity[device_id][variant];
    slot->cg_parity = g_cn_cg_parity[device_id][variant];
    slot->hardened_mode = g_cn_hardened_mode[device_id][variant];
    slot->hardened_threads = g_cn_hardened_threads[device_id][variant];
    slot->block_threads = g_cn_block_tuned_threads[device_id][variant];
    slot->setup_mode = g_cn_word_setup[device_id][variant];
    slot->final_mode = g_cn_word_final[device_id][variant];
    slot->overlap_mode = g_cn_overlap_mode[device_id][variant];
    slot->stagger_mode = g_cn_stagger_mode[device_id][variant];
    slot->overlap_reported = g_cn_overlap_reported[device_id][variant];
    slot->stagger_reported = g_cn_stagger_reported[device_id][variant];
}

void gpu_variant_runtime_restore(int device_id,
                                 std::size_t variant,
                                 const GpuVariantRuntimeBatchSlot& slot)
{
    g_cn_production_selector[device_id][variant] = slot.selector;
    g_cn_production_geometry[device_id][variant] = slot.geometry;
    g_cn_residency_mode[device_id][variant] = slot.residency_mode;
    g_cn_residency_threads[device_id][variant] = slot.residency_threads;
    g_cn_ttable4_parity[device_id][variant] = slot.ttable4_parity;
    g_cn_tile64_parity[device_id][variant] = slot.tile64_parity;
    g_cn_cg_parity[device_id][variant] = slot.cg_parity;
    g_cn_hardened_mode[device_id][variant] = slot.hardened_mode;
    g_cn_hardened_threads[device_id][variant] = slot.hardened_threads;
    g_cn_block_tuned_threads[device_id][variant] = slot.block_threads;
    g_cn_word_setup[device_id][variant] = slot.setup_mode;
    g_cn_word_final[device_id][variant] = slot.final_mode;
    g_cn_overlap_mode[device_id][variant] = slot.overlap_mode;
    g_cn_stagger_mode[device_id][variant] = slot.stagger_mode;
    g_cn_overlap_reported[device_id][variant] = slot.overlap_reported;
    g_cn_stagger_reported[device_id][variant] = slot.stagger_reported;
}

void gpu_variant_runtime_reset(int device_id, std::size_t variant)
{
    g_cn_production_selector[device_id][variant] = CnProductionSelectorState{};
    g_cn_production_geometry[device_id][variant] = CnProductionGeometryState{};
    g_cn_residency_mode[device_id][variant] = 0;
    g_cn_residency_threads[device_id][variant] = 0;
    g_cn_ttable4_parity[device_id][variant] = 0;
    g_cn_tile64_parity[device_id][variant] = 0;
    g_cn_cg_parity[device_id][variant] = 0;
    g_cn_hardened_mode[device_id][variant] = 0;
    g_cn_hardened_threads[device_id][variant] = 0;
    g_cn_block_tuned_threads[device_id][variant] = 0;
    g_cn_word_setup[device_id][variant] = 0;
    g_cn_word_final[device_id][variant] = 0;
    g_cn_overlap_mode[device_id][variant] = 0;
    g_cn_stagger_mode[device_id][variant] = 0;
    g_cn_overlap_reported[device_id][variant] = false;
    g_cn_stagger_reported[device_id][variant] = false;
}

void gpu_variant_activate_runtime_batch(
    int device_id,
    std::size_t count,
    const std::array<bool, kGpuVariantBatchCount>& used_variants)
{
    if (device_id < 0 || device_id >= kCn2LaneMaxDevices || count == 0U) return;

    for (std::size_t variant = 0; variant < used_variants.size(); ++variant) {
        if (!used_variants[variant]) continue;

        auto& active_count = g_gpu_variant_runtime_active_count[device_id][variant];
        if (active_count == count) continue;

        const std::size_t previous = active_count;
        if (previous != 0U)
            gpu_variant_runtime_save_active(device_id, variant, previous);

        const auto* saved = gpu_variant_runtime_find_slot(device_id, variant, count);
        if (saved) {
            gpu_variant_runtime_restore(device_id, variant, *saved);
        } else {
            gpu_variant_runtime_reset(device_id, variant);
        }
        active_count = count;

        if (previous != 0U) {
            std::cout << "[GPU " << device_id << "] CN runtime batch state"
                      << " | variant=" << cryptonight::kVariantConfigs[variant].name
                      << " | batch=" << previous << " -> " << count
                      << " | source=" << (saved ? "memory-cache" : "fresh")
                      << '\n';
        }
    }
}

} // anonymous namespace
