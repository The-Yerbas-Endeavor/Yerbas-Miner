#pragma once

#include <cstddef>
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
    // 16 hashes per worker better amortizes scheduler/thread-launch overhead
    // while keeping Stratum job switching responsive on the reference backend.
    unsigned int cpu_batch{16};
};

struct GpuConfig {
    bool enabled{true};
    // Empty means use every detected CUDA device. This is safer than assuming
    // that device 0 is always the GPU the user intended to mine with.
    std::vector<int> devices{};
    int intensity{0};
    // Explicit production CUDA batch size. 0 keeps legacy auto behavior.
    // Pascal/GTX 1080 Ti tuning currently favors 3584 hashes per batch.
    std::size_t batch_size{0};
    // Skip the startup CUDA readiness probe after the backend has already been
    // validated on the machine. Full mining coverage is still required.
    bool skip_validation{false};
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
