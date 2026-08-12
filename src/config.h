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
};

struct GpuConfig {
    bool enabled{true};
    std::vector<int> devices{0};
    int intensity{0};
};

struct LoggingConfig {
    std::string level{"info"};
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
