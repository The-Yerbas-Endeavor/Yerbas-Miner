#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "hash_selection.h"
#include "uint256.h"

namespace yerbas::ghostrider {

Hash256 hash_reference(const Work& work)
{
    if (work.data == nullptr || work.size == 0) {
        throw std::invalid_argument("GhostRider work buffer is empty");
    }

    uint256 prev_block_hash;
    std::memcpy(prev_block_hash.begin(), work.prev_block_hash.data(), work.prev_block_hash.size());

    uint512 hash[18];
    HashSelection hash_selection(
        prev_block_hash,
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
        {0, 1, 2, 3, 4, 5});

    const std::vector<int> random_cns(hash_selection.getCnIndexes());
    const std::vector<int> core_hash_indexes(hash_selection.getAlgoIndexes());

    for (int i = 0; i < 18; ++i) {
        const void* to_hash = nullptr;
        int len_to_hash = 0;

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

        // These are the exact Yerbas Core GhostRider primitives. Each stage calls
        // both functions; the selector value of -1 makes the non-selected switch
        // fall through without modifying the stage output.
        coreHash(to_hash, &hash[i], len_to_hash, core_selection);
        cnHash(&hash[i - 1], &hash[i], len_to_hash, cn_selection);
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
