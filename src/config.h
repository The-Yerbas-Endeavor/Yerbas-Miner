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
    // Hidden config-only testing switch. Production default keeps the
    // developer fee enabled; config.json may set "developer_fee": false.
    bool developer_fee{true};
    // Explicit combined calibration flag. Forces fresh CPU tuning and pairs
    // with GPU autotune when --autotune is requested.
    bool autotune{false};
};

struct GpuConfig {
    bool enabled{true};
    std::vector<int> devices{};
    int intensity{0};
    bool skip_validation{false};
    // User-facing GPU tuning policy:
    //   auto = use saved tuning; first-run calibration when needed
    //   full = force fresh bounded calibration plus deep production retune
    //   off  = never request GPU benchmarking; use saved/safe production state
    std::string gpu_tune{"auto"};
    // Internal/legacy one-shot bounded calibration request. First-run setup and
    // --autotune use this without implicitly enabling the deep production retune.
    bool autotune{false};
};

struct LoggingConfig {
    std::string level{"info"};

    // Console presentation:
    //   auto  = single-screen dashboard on an interactive terminal, plain otherwise
    //   tui   = request the single-screen dashboard
    //   plain = traditional scrolling console
    std::string console_mode{"auto"};

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
