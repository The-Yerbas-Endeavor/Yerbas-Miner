#include "config.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace yerbas {
namespace {

using json = nlohmann::json;

std::string require_value(int argc, char** argv, int& i, const char* option)
{
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + option);
    }
    return argv[++i];
}

std::vector<int> parse_devices(const std::string& value)
{
    std::vector<int> devices;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = value.find(',', start);
        const std::string token = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            devices.push_back(std::stoi(token));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (devices.empty()) {
        throw std::runtime_error("--devices requires at least one device id");
    }
    return devices;
}

void apply_json(AppConfig& cfg, const json& root)
{
    if (root.contains("pool")) {
        const auto& pool = root.at("pool");
        if (pool.contains("url")) cfg.pool.url = pool.at("url").get<std::string>();
        if (pool.contains("user")) cfg.pool.user = pool.at("user").get<std::string>();
        if (pool.contains("password")) cfg.pool.password = pool.at("password").get<std::string>();
    }
    if (root.contains("miner")) {
        const auto& miner = root.at("miner");
        if (miner.contains("worker")) cfg.miner.worker = miner.at("worker").get<std::string>();
        if (miner.contains("threads")) cfg.miner.threads = miner.at("threads").get<unsigned int>();
    }
    if (root.contains("gpu")) {
        const auto& gpu = root.at("gpu");
        if (gpu.contains("enabled")) cfg.gpu.enabled = gpu.at("enabled").get<bool>();
        if (gpu.contains("devices")) cfg.gpu.devices = gpu.at("devices").get<std::vector<int>>();
        if (gpu.contains("intensity")) cfg.gpu.intensity = gpu.at("intensity").get<int>();
    }
    if (root.contains("logging")) {
        const auto& logging = root.at("logging");
        if (logging.contains("level")) cfg.logging.level = logging.at("level").get<std::string>();
    }
}

void load_json_file(AppConfig& cfg)
{
    if (!std::filesystem::exists(cfg.config_path)) {
        return;
    }

    std::ifstream input(cfg.config_path);
    if (!input) {
        throw std::runtime_error("Unable to open config file: " + cfg.config_path);
    }

    json root;
    input >> root;
    apply_json(cfg, root);
}

} // namespace

AppConfig load_config(int argc, char** argv)
{
    AppConfig cfg;

    // First pass only discovers the config path/help so JSON loads before CLI overrides.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            cfg.config_path = require_value(argc, argv, i, "--config");
        } else if (arg == "--help" || arg == "-h") {
            print_config_help(argv[0]);
            std::exit(0);
        }
    }

    load_json_file(cfg);

    // CLI values intentionally override config.json values.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            ++i;
        } else if (arg == "--pool") {
            cfg.pool.url = require_value(argc, argv, i, "--pool");
        } else if (arg == "--user") {
            cfg.pool.user = require_value(argc, argv, i, "--user");
        } else if (arg == "--password") {
            cfg.pool.password = require_value(argc, argv, i, "--password");
        } else if (arg == "--worker") {
            cfg.miner.worker = require_value(argc, argv, i, "--worker");
        } else if (arg == "--threads") {
            cfg.miner.threads = static_cast<unsigned int>(std::stoul(require_value(argc, argv, i, "--threads")));
        } else if (arg == "--devices") {
            cfg.gpu.devices = parse_devices(require_value(argc, argv, i, "--devices"));
        } else if (arg == "--intensity") {
            cfg.gpu.intensity = std::stoi(require_value(argc, argv, i, "--intensity"));
        } else if (arg == "--no-gpu") {
            cfg.gpu.enabled = false;
        } else if (arg == "--log-level") {
            cfg.logging.level = require_value(argc, argv, i, "--log-level");
        } else if (arg == "--help" || arg == "-h") {
            // handled on pass one
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    return cfg;
}

void print_config_help(const char* program)
{
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  --config FILE       Config file (default: config.json)\n"
        << "  --pool URL          Stratum pool URL\n"
        << "  --user USER         Wallet/address or pool username\n"
        << "  --password PASS     Pool password (default: x)\n"
        << "  --worker NAME       Worker name\n"
        << "  --threads N         CPU/reference worker threads (0 = auto)\n"
        << "  --devices 0,1       GPU device ids\n"
        << "  --intensity N       GPU intensity (0 = auto)\n"
        << "  --no-gpu            Disable GPU backend\n"
        << "  --log-level LEVEL   debug, info, warn, error\n"
        << "  -h, --help          Show this help\n";
}

} // namespace yerbas
