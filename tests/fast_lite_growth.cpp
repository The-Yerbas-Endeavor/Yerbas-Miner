#include "stratum/fast_lite_growth.h"
#include <iostream>
#include <limits>

int main()
{
    using namespace yerbas::stratum;
    int failures = 0;
    auto check = [&failures](bool ok, const char* name) {
        if (!ok) { std::cerr << "FAIL: " << name << '\n'; ++failures; }
    };
    check(fast_lite_growth_target(3584, 4096) == 4096, "8 GiB budget");
    check(fast_lite_growth_target(3584, 4480) == 4480, "8960 MiB budget");
    check(fast_lite_growth_target(3584, 10000) == 4480, "25 percent ceiling");
    check(fast_lite_growth_target(3584, 3500) == 3584, "no headroom");
    check(fast_lite_growth_target(3584, 3600) == 3584, "alignment without growth");
    check(fast_lite_growth_target(0, 4096) == 0, "empty baseline");
    const auto max = std::numeric_limits<std::size_t>::max();
    check(fast_lite_growth_target(max - 1024, max) >= max - 1024, "overflow guard");
    check(fast_lite_growth_target(8192, 12000) == 10240, "generic larger baseline");
    for (std::size_t base = 128; base <= 8192; base += 128) {
        for (std::size_t safe = base; safe <= 16384; safe += 127) {
            const auto target = fast_lite_growth_target(base, safe);
            check(target >= base && target <= safe && target <= base + base / 4,
                  "growth always respects baseline, allocation, and ceiling");
        }
    }
    FastLiteGrowthState state{};
    check(!state.ready(false, true, 60), "other CN classes excluded");
    check(!state.ready(true, false, 60) && state.clean_scans == 0, "stale/shaped excluded");
    check(!state.ready(true, true, 60), "first clean scan");
    check(!state.ready(true, true, 60), "second clean scan");
    check(!state.ready(true, true, 29.99), "young job excluded");
    check(state.ready(true, true, 30), "stable job eligible");
    check(!state.ready(true, true, 60), "one attempt per job");
    state = {};
    check(!state.ready(true, true, 60), "new job restarts qualification");
    if (failures) return 1;
    std::cout << "Fast/Lite growth policy checks passed\n";
    return 0;
}
