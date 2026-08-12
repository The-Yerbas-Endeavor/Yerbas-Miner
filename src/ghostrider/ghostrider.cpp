#include "ghostrider/ghostrider.h"

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

void selections(const Work& work,
                std::vector<int>& core_hash_indexes,
                std::vector<int>& random_cns)
{
    HashSelection hash_selection(
        previous_block_hash(work),
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
        {0, 1, 2, 3, 4, 5});
    random_cns = hash_selection.getCnIndexes();
    core_hash_indexes = hash_selection.getAlgoIndexes();
}
}

StageSchedule stage_schedule(const Work& work)
{
    std::vector<int> core_hash_indexes;
    std::vector<int> random_cns;
    selections(work, core_hash_indexes, random_cns);

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

Hash256 hash_reference(const Work& work)
{
    if (work.size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GhostRider work buffer is too large");
    }

    std::vector<int> core_hash_indexes;
    std::vector<int> random_cns;
    selections(work, core_hash_indexes, random_cns);

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

        int core_selection = -1;
        int cn_selection = -1;

        if (i < 5) {
            core_selection = core_hash_indexes[i];
        } else if (i == 5) {
            cn_selection = random_cns[0];
        } else if (i < 11) {
            core_selection = core_hash_indexes[i - 1];
        } else if (i == 11) {
            cn_selection = random_cns[1];
        } else if (i < 17) {
            core_selection = core_hash_indexes[i - 2];
        } else {
            cn_selection = random_cns[2];
        }

        coreHash(to_hash, &hash[i], len_to_hash, core_selection);
        uint512* cn_input = (i == 0) ? &hash[0] : &hash[i - 1];
        cnHash(cn_input, &hash[i], len_to_hash, cn_selection);
    }

    const uint256 result = hash[17].trim256();
    Hash256 out{};
    std::memcpy(out.data(), result.begin(), out.size());
    return out;
}

bool reference_ready() noexcept
{
    return true;
}

} // namespace yerbas::ghostrider
