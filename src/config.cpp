#include "config.h"
#include "cpu/cpu_autotune.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

namespace yerbas {
namespace {
using json = nlohmann::json;

std::string require_value(int argc, char** argv, int& i, const char* option)
{
    if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + option);
    return argv[++i];
}

std::vector<int> parse_devices(const std::string& value)
{
    std::vector<int> devices;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = value.find(',', start);
        const std::string token = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) devices.push_back(std::stoi(token));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (devices.empty()) throw std::runtime_error("--devices requires at least one device id");
    return devices;
}

void apply_json(AppConfig& cfg, const json& root)
{
    if (root.contains("pool")) {
        const auto& p = root.at("pool");
        if (p.contains("url")) cfg.pool.url = p.at("url").get<std::string>();
        if (p.contains("user")) cfg.pool.user = p.at("user").get<std::string>();
        if (p.contains("password")) cfg.pool.password = p.at("password").get<std::string>();
    }
    if (root.contains("miner")) {
        const auto& m = root.at("miner");
        if (m.contains("worker")) cfg.miner.worker = m.at("worker").get<std::string>();
        if (m.contains("cpu_enabled")) cfg.miner.cpu_enabled = m.at("cpu_enabled").get<bool>();
        if (m.contains("threads")) cfg.miner.threads = m.at("threads").get<unsigned int>();
        if (m.contains("hybrid")) cfg.miner.hybrid = m.at("hybrid").get<bool>();
        if (m.contains("cpu_batch")) cfg.miner.cpu_batch = m.at("cpu_batch").get<unsigned int>();
    }
    if (root.contains("gpu")) {
        const auto& g = root.at("gpu");
        if (g.contains("enabled")) cfg.gpu.enabled = g.at("enabled").get<bool>();
        if (g.contains("devices")) cfg.gpu.devices = g.at("devices").get<std::vector<int>>();
        if (g.contains("intensity")) cfg.gpu.intensity = g.at("intensity").get<int>();
        if (g.contains("skip_validation")) cfg.gpu.skip_validation = g.at("skip_validation").get<bool>();
    }
    if (root.contains("logging")) {
        const auto& l = root.at("logging");
        if (l.contains("level")) cfg.logging.level = l.at("level").get<std::string>();
    }
}

void load_json_file(AppConfig& cfg)
{
    if (!std::filesystem::exists(cfg.config_path)) return;
    std::ifstream input(cfg.config_path);
    if (!input) throw std::runtime_error("Unable to open config file: " + cfg.config_path);
    json root; input >> root; apply_json(cfg, root);
}
}

AppConfig load_config(int argc, char** argv)
{
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") cfg.config_path = require_value(argc, argv, i, "--config");
        else if (arg == "--help" || arg == "-h") { print_config_help(argv[0]); std::exit(0); }
    }
    load_json_file(cfg);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") ++i;
        else if (arg == "--pool") cfg.pool.url = require_value(argc, argv, i, "--pool");
        else if (arg == "--user") cfg.pool.user = require_value(argc, argv, i, "--user");
        else if (arg == "--password") cfg.pool.password = require_value(argc, argv, i, "--password");
        else if (arg == "--worker") cfg.miner.worker = require_value(argc, argv, i, "--worker");
        else if (arg == "--threads") cfg.miner.threads = static_cast<unsigned int>(std::stoul(require_value(argc, argv, i, "--threads")));
        else if (arg == "--cpu-batch") cfg.miner.cpu_batch = static_cast<unsigned int>(std::stoul(require_value(argc, argv, i, "--cpu-batch")));
        else if (arg == "--no-cpu") cfg.miner.cpu_enabled = false;
        else if (arg == "--no-hybrid") cfg.miner.hybrid = false;
        else if (arg == "--devices") cfg.gpu.devices = parse_devices(require_value(argc, argv, i, "--devices"));
        else if (arg == "--intensity") cfg.gpu.intensity = std::stoi(require_value(argc, argv, i, "--intensity"));
        else if (arg == "--no-gpu") cfg.gpu.enabled = false;
        else if (arg == "--skip-validation") cfg.gpu.skip_validation = true;
        else if (arg == "--log-level") cfg.logging.level = require_value(argc, argv, i, "--log-level");
        else if (arg == "--help" || arg == "-h") {}
        else throw std::runtime_error("Unknown option: " + arg);
    }

    if (cfg.miner.cpu_batch == 0) cfg.miner.cpu_batch = 16;

    // A configured thread count is a ceiling for production autotuning, not an
    // assumption that every logical worker improves GhostRider throughput.
    // Skip the benchmark for incomplete pool configs so setup/help remains fast.
    if (cfg.miner.cpu_enabled && !cfg.pool.url.empty() && !cfg.pool.user.empty()) {
        const unsigned int hardware_threads = std::max(1U, std::thread::hardware_concurrency());
        const auto tune = cpu::production_autotune(hardware_threads,
                                                   cfg.miner.threads,
                                                   cfg.miner.cpu_batch);
        cfg.miner.threads = tune.threads;
        cfg.miner.cpu_batch = tune.batch;
    }
    return cfg;
}

void print_config_help(const char* program)
{
    std::cout << "Usage: " << program << " [options]\n\n"
        << "  --config FILE       Config file (default: config.json)\n"
        << "  --pool URL          Stratum pool URL\n"
        << "  --user USER         Wallet/address or pool username\n"
        << "  --password PASS     Pool password (default: x)\n"
        << "  --worker NAME       Worker name\n"
        << "  --threads N         CPU thread ceiling (0 = all logical CPUs, autotuned)\n"
        << "  --cpu-batch N       CPU autotune starting batch (0 = default 16)\n"
        << "  --no-cpu            Disable CPU mining\n"
        << "  --no-hybrid         Do not combine CPU and GPU\n"
        << "  --devices 0,1       GPU device ids\n"
        << "  --intensity N       GPU intensity (0 = auto)\n"
        << "  --no-gpu            Disable GPU backend\n"
        << "  --skip-validation   Skip startup CUDA readiness probe\n"
        << "  --log-level LEVEL   debug, info, warn, error\n"
        << "  -h, --help          Show this help\n\n"
        << "CPU autotune environment:\n"
        << "  YERBAS_CPU_RETUNE=1            Ignore cached CPU tuning and benchmark again\n"
        << "  YERBAS_CPU_DISABLE_AUTOTUNE=1  Use configured/default CPU threads and batch\n"
        << "  YERBAS_DIAGNOSTICS=1          Show individual CPU autotune benchmark results\n";
}

} // namespace yerbas
