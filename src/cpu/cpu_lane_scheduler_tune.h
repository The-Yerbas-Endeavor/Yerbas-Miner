#pragma once

#include <string>

namespace yerbas::cpu {

struct LaneSchedulerTuneResult {
    unsigned int workers{1};
    unsigned int lanes{1};
    double throughput_hps{0.0};
    bool from_cache{false};
};

LaneSchedulerTuneResult tune_lane_scheduler(unsigned int hardware_threads,
                                            unsigned int configured_threads,
                                            const std::string& mode);

} // namespace yerbas::cpu
