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
    unsigned int cpu_batch{0};
    unsigned int cpu_lanes{1};
    std::string cpu_tune{"default"};
    // Explicit combined calibration flag. Forces fresh CPU tuning and pairs
    // with GPU autotune when --autotune is requested.
    bool autotune{false};
};

struct GpuConfig {
    bool enabled{true};
    std::vector<int> devices{};
    int intensity{0};
    bool skip_validation{false};
    // Explicit one-shot calibration. Normal production startup never benchmarks.
    bool autotune{false};
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
