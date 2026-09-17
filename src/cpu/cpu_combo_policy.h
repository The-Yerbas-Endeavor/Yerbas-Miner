#pragma once

#include "ghostrider/ghostrider.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace yerbas::cpu_combo_policy {

using WidthPolicy = std::array<unsigned int, 6>;

struct Rule {
    std::uint32_t mask{0U};
    WidthPolicy widths{{1U, 1U, 1U, 1U, 1U, 1U}};
    double confirmed_gain_pct{0.0};
};

struct Policy {
    bool enabled{false};
    bool valid{false};
    std::string path;
    unsigned int workers{0U};
    unsigned int batch{0U};
    std::string affinity;
    WidthPolicy baseline{{1U, 1U, 1U, 1U, 1U, 1U}};
    std::vector<Rule> rules;
};

inline bool valid_width(unsigned int width) noexcept
{
    return width == 1U || width == 2U || width == 4U;
}

inline std::uint32_t combination_mask(const ghostrider::StageSchedule& schedule) noexcept
{
    std::uint32_t mask = 0U;
    for (const std::uint8_t stage : schedule) {
        if ((stage & ghostrider::kCryptoNightStageFlag) == 0U) continue;
        const auto variant = static_cast<unsigned int>(stage & 0x7fU);
        if (variant < 6U) mask |= (1U << variant);
    }
    return mask;
}

inline Policy load_policy()
{
    Policy out{};
    const char* env = std::getenv("YERBAS_CPU_COMBO_POLICY_FILE");
    if (env == nullptr || *env == '\0') return out;

    out.enabled = true;
    out.path = env;

    std::ifstream in(out.path);
    if (!in) {
        std::cerr << "[CPU combo policy] could not open: " << out.path << '\n';
        return out;
    }

    std::string magic;
    unsigned int revision = 0U;
    if (!(in >> magic >> revision) || magic != "YERBAS_CPU_COMBO_POLICY" || revision != 1U) {
        std::cerr << "[CPU combo policy] invalid header: " << out.path << '\n';
        return out;
    }

    std::string token;
    while (in >> token) {
        if (token == "BASE") {
            if (!(in >> out.workers >> out.batch >> out.affinity)) return out;
            for (auto& width : out.baseline) {
                if (!(in >> width) || !valid_width(width)) return out;
            }
            continue;
        }

        if (token == "RULE") {
            Rule rule{};
            if (!(in >> rule.mask)) return out;
            for (auto& width : rule.widths) {
                if (!(in >> width) || !valid_width(width)) return out;
            }
            if (!(in >> rule.confirmed_gain_pct)) return out;
            if (rule.mask == 0U || rule.confirmed_gain_pct < 0.0) return out;
            out.rules.push_back(rule);
            continue;
        }

        std::string rest;
        std::getline(in, rest);
    }

    out.valid = out.workers > 0U && out.batch > 0U && !out.rules.empty();
    if (out.valid) {
        std::cout << "[CPU combo policy] loaded | file=" << out.path
                  << " | rules=" << out.rules.size()
                  << " | baseline="
                  << out.baseline[0] << '/' << out.baseline[1] << '/'
                  << out.baseline[2] << '/' << out.baseline[3] << '/'
                  << out.baseline[4] << '/' << out.baseline[5]
                  << " | experimental=yes\n";
    } else {
        std::cerr << "[CPU combo policy] invalid/incomplete file: " << out.path << '\n';
    }
    return out;
}

inline const Policy& policy()
{
    static const Policy cached = load_policy();
    return cached;
}

inline WidthPolicy select(const ghostrider::StageSchedule& schedule,
                          const WidthPolicy& baseline) noexcept
{
    const Policy& p = policy();
    if (!p.enabled || !p.valid || p.baseline != baseline) return baseline;

    const std::uint32_t mask = combination_mask(schedule);
    for (const auto& rule : p.rules) {
        if (rule.mask == mask) return rule.widths;
    }
    return baseline;
}

} // namespace yerbas::cpu_combo_policy
