#pragma once

#include <array>
#include <string>

namespace yerbas::cpu {

struct CnWidthTuneResult {
    std::array<unsigned int, 6> widths{{1,1,1,1,1,1}};
    bool from_cache{false};
};

CnWidthTuneResult qualify_cn_widths(const std::string& mode);

} // namespace yerbas::cpu
