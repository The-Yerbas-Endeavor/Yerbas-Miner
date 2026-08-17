#include "ghostrider/ghostrider.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hash_selection.h"
#include "uint256.h"

namespace yerbas::ghostrider {

namespace {
constexpr std::size_t kVersionSize = 4;
constexpr std::size_t kPrevHashSize = 32;
constexpr std::size_t kMinimumHeaderPrefix = kVersionSize + kPrevHashSize;

uint256 previous_block_hash(const Work& work)
{
    if (work.data == nullptr || work.size < kMinimumHeaderPrefix) {
        throw std::invalid_argument(
            "GhostRider input must contain at least nVersion and hashPrevBlock");
    }

    uint256 prev_block_hash;
    std::memcpy(prev_block_hash.begin(), work.data + kVersionSize, kPrevHashSize);
    return prev_block_hash;
}

StageSchedule build_stage_schedule(const Work& work)
{
    HashSelection hash_selection(
        previous_block_hash(work),
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
        {0, 1, 2, 3, 4, 5});

    const std::vector<int> random_cns = hash_selection.getCnIndexes();
    const std::vector<int> core_hash_indexes = hash_selection.getAlgoIndexes();

    StageSchedule schedule{};
    for (int i = 0; i < 18; ++i) {
        if (i < 5) {
            schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core_hash_indexes[i]);
        } else if (i == 5) {
            schedule[5] = static_cast<std::uint8_t>(kCryptoNightStageFlag | random_cns[0]);
        } else if (i < 11) {
            schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core_hash_indexes[i - 1]);
        } else if (i == 11) {
            schedule[11] = static_cast<std::uint8_t>(kCryptoNightStageFlag | random_cns[1]);
        } else if (i < 17) {
            schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core_hash_indexes[i - 2]);
        } else {
            schedule[17] = static_cast<std::uint8_t>(kCryptoNightStageFlag | random_cns[2]);
        }
    }
    return schedule;
}

const StageSchedule& cached_stage_schedule(const Work& work)
{
    if (work.data == nullptr || work.size < kMinimumHeaderPrefix) {
        throw std::invalid_argument(
            "GhostRider input must contain at least nVersion and hashPrevBlock");
    }

    struct ThreadScheduleCache {
        std::array<std::uint8_t, kPrevHashSize> prev_hash{};
        StageSchedule schedule{};
        bool valid{false};
    };

    thread_local ThreadScheduleCache cache;
    const auto* prev = work.data + kVersionSize;
    if (!cache.valid || std::memcmp(cache.prev_hash.data(), prev, kPrevHashSize) != 0) {
        std::memcpy(cache.prev_hash.data(), prev, kPrevHashSize);
        cache.schedule = build_stage_schedule(work);
        cache.valid = true;
    }
    return cache.schedule;
}

Hash256 hash_with_schedule(const Work& work, const StageSchedule& schedule)
{
    uint512 hash[18];
    for (int i = 0; i < 18; ++i) {
        const void* to_hash;
        int len_to_hash;

        if (i == 0) {
            to_hash = static_cast<const void*>(work.data);
            len_to_hash = static_cast<int>(work.size);
        } else {
            to_hash = static_cast<const void*>(&hash[i - 1]);
            len_to_hash = 64;
        }

        const std::uint8_t stage = schedule[static_cast<std::size_t>(i)];
        int core_selection = -1;
        int cn_selection = -1;
        if ((stage & kCryptoNightStageFlag) != 0) {
            cn_selection = static_cast<int>(stage & 0x7fU);
        } else {
            core_selection = static_cast<int>(stage);
        }

        // Preserve Yerbas Core's exact reference call sequence. The inactive
        // selector is -1 and the corresponding helper is a no-op.
        coreHash(to_hash, &hash[i], len_to_hash, core_selection);
        uint512* cn_input = (i == 0) ? &hash[0] : &hash[i - 1];
        cnHash(cn_input, &hash[i], len_to_hash, cn_selection);
    }

    const uint256 result = hash[17].trim256();
    Hash256 out{};
    std::memcpy(out.data(), result.begin(), out.size());
    return out;
}
} // namespace

Hash512 core_hash_reference(const Work& work, int algorithm)
{
    if (work.data == nullptr || work.size == 0) {
        throw std::invalid_argument("Core hash input must not be empty");
    }
    if (work.size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Core hash work buffer is too large");
    }
    if (algorithm < 0 || algorithm > 14) {
        throw std::invalid_argument("Core hash algorithm index must be 0..14");
    }

    uint512 result;
    coreHash(work.data, &result, static_cast<int>(work.size), algorithm);

    Hash512 out{};
    std::memcpy(out.data(), result.begin(), out.size());
    return out;
}

StageSchedule stage_schedule(const Work& work)
{
    return build_stage_schedule(work);
}

Hash256 hash_reference(const Work& work)
{
    if (work.data == nullptr || work.size == 0) {
        throw std::invalid_argument("GhostRider work buffer must not be empty");
    }
    if (work.size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GhostRider work buffer is too large");
    }

    return hash_with_schedule(work, cached_stage_schedule(work));
}

bool reference_ready() noexcept
{
    return true;
}

} // namespace yerbas::ghostrider
