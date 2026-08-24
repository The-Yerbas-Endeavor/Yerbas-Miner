#pragma once

#include <array>
#include <string>

namespace yerbas::cpu {

struct CnWidthTuneResult {
    std::array<unsigned int, 6> widths{{1,1,1,1,1,1}};
    bool from_cache{false};
};

CnWidthTuneResult qualify_cn_widths(const std::string& mode);

// Returns the width plan selected/loaded during startup qualification.
// Defaults to all 1-way when tuning is off or has not run.
std::array<unsigned int, 6> active_cn_widths() noexcept;

} // namespace yerbas::cpu
