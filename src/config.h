#pragma once

#include <string>
#include <vector>

namespace yerbas {

struct PoolConfig {
    std::string url;
    std::string user;
    std::string password{"x"};
};

struct MinerConfig {
    std::string worker{"rig1"};
    bool cpu_enabled{true};
    unsigned int threads{0};
    bool hybrid{true};
    unsigned int cpu_batch{16};
    unsigned int cpu_lanes{1};
    std::string cpu_tune{"off"};
};

struct GpuConfig {
    bool enabled{true};
    std::vector<int> devices{};
    int intensity{0};
    bool skip_validation{false};
    // Force a visible production-batch CN-Fast backend-family benchmark at
    // startup. The CN-Fast family cache is bypassed for this run and normal
    // mining continues with the parity-qualified winner.
    bool benchmark_cn_fast{false};
};

struct LoggingConfig {
    std::string level{"info"};
    // Optional rotation/performance CSV. Empty disables file logging.
    std::string perf_csv;
};

struct AppConfig {
    PoolConfig pool;
    MinerConfig miner;
    GpuConfig gpu;
    LoggingConfig logging;
    std::string config_path{"config.json"};
};

AppConfig load_config(int argc, char** argv);
void print_config_help(const char* program);

} // namespace yerbas
