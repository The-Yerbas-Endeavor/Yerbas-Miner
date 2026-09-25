#include "config.h"

#include <algorithm>
#include <cctype>
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

std::string normalize_tune_mode(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "none" || value == "no" || value == "false" || value == "0") value = "off";
    if (value != "off" && value != "simple" && value != "default" && value != "full")
        throw std::runtime_error("CPU tune mode must be off, simple, default, or full");
    return value;
}

std::string normalize_gpu_tune_mode(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "none" || value == "no" || value == "false" || value == "0") value = "off";
    if (value == "yes" || value == "true" || value == "1") value = "full";
    if (value != "off" && value != "auto" && value != "full")
        throw std::runtime_error("GPU tune mode must be off, auto, or full");
    return value;
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
        if (m.contains("cpu_tune")) cfg.miner.cpu_tune = normalize_tune_mode(m.at("cpu_tune").get<std::string>());
        if (m.contains("developer_fee")) cfg.miner.developer_fee = m.at("developer_fee").get<bool>();
    }
    if (root.contains("gpu")) {
        const auto& g = root.at("gpu");
        if (g.contains("enabled")) cfg.gpu.enabled = g.at("enabled").get<bool>();
        if (g.contains("devices")) cfg.gpu.devices = g.at("devices").get<std::vector<int>>();
        if (g.contains("intensity")) cfg.gpu.intensity = g.at("intensity").get<int>();
        if (g.contains("skip_validation")) cfg.gpu.skip_validation = g.at("skip_validation").get<bool>();
        if (g.contains("gpu_tune")) cfg.gpu.gpu_tune = normalize_gpu_tune_mode(g.at("gpu_tune").get<std::string>());
        else if (g.contains("tune")) cfg.gpu.gpu_tune = normalize_gpu_tune_mode(g.at("tune").get<std::string>());
        // Legacy config compatibility. The old boolean requested the bounded
        // calibration pass, not the newer deep production retune.
        if (g.contains("autotune")) cfg.gpu.autotune = g.at("autotune").get<bool>();
    }
    if (root.contains("logging")) {
        const auto& l = root.at("logging");
        if (l.contains("level")) cfg.logging.level = l.at("level").get<std::string>();
        if (l.contains("console")) cfg.logging.console_mode = l.at("console").get<std::string>();
        if (l.contains("perf_csv")) cfg.logging.perf_csv = l.at("perf_csv").get<std::string>();
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
        else if (arg == "--tune") cfg.miner.cpu_tune = normalize_tune_mode(require_value(argc, argv, i, "--tune"));
        else if (arg == "--autotune") {
            cfg.miner.autotune = true;
            cfg.miner.cpu_tune = "default";
            cfg.gpu.autotune = true;
        }
        else if (arg == "--no-tune") cfg.miner.cpu_tune = "off";
        else if (arg == "--no-cpu") cfg.miner.cpu_enabled = false;
        else if (arg == "--no-hybrid") cfg.miner.hybrid = false;
        else if (arg == "--devices") cfg.gpu.devices = parse_devices(require_value(argc, argv, i, "--devices"));
        else if (arg == "--intensity") cfg.gpu.intensity = std::stoi(require_value(argc, argv, i, "--intensity"));
        else if (arg == "--gpu-tune") cfg.gpu.gpu_tune = normalize_gpu_tune_mode(require_value(argc, argv, i, "--gpu-tune"));
        else if (arg == "--gpu-autotune") cfg.gpu.autotune = true;
        else if (arg == "--no-gpu") cfg.gpu.enabled = false;
        else if (arg == "--skip-validation") cfg.gpu.skip_validation = true;
        else if (arg == "--log-level") cfg.logging.level = require_value(argc, argv, i, "--log-level");
        else if (arg == "--console") cfg.logging.console_mode = require_value(argc, argv, i, "--console");
        else if (arg == "--perf-log") cfg.logging.perf_csv = require_value(argc, argv, i, "--perf-log");
        else if (arg == "--help" || arg == "-h") {}
        else throw std::runtime_error("Unknown option: " + arg);
    }

    cfg.miner.cpu_tune = normalize_tune_mode(cfg.miner.cpu_tune);
    cfg.gpu.gpu_tune = normalize_gpu_tune_mode(cfg.gpu.gpu_tune);

    if (cfg.logging.console_mode != "auto" &&
        cfg.logging.console_mode != "tui" &&
        cfg.logging.console_mode != "plain") {
        throw std::runtime_error(
            "Invalid console mode: " + cfg.logging.console_mode +
            " (expected auto, tui, or plain)");
    }

    if (cfg.miner.cpu_batch == 0) cfg.miner.cpu_batch = 16;
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
        << "  --threads N         CPU thread ceiling (0 = all logical CPUs)\n"
        << "  --cpu-batch N       CPU batch when tuning is off / initial reference value\n"
        << "  --tune MODE         CPU tuning: off, simple, default, full\n"
        << "  --autotune          Fresh bounded CPU + GPU calibration with visible progress\n"
        << "  --no-tune           Start immediately with configured/default CPU settings\n"
        << "  --no-cpu            Disable CPU mining\n"
        << "  --no-hybrid         Do not combine CPU and GPU\n"
        << "  --devices 0,1       GPU device ids\n"
        << "  --intensity N       GPU intensity (0 = auto)\n"
        << "  --gpu-tune MODE     GPU tuning policy: off, auto, full\n"
        << "  --gpu-autotune      Run one bounded GPU calibration and cache the result\n"
        << "  --no-gpu            Disable GPU backend\n"
        << "  --skip-validation   Skip startup CUDA readiness probe\n"
        << "  --log-level LEVEL   debug, info, warn, error\n"
        << "  --console MODE      auto, tui, plain\n"
        << "  --perf-log FILE     Append rotation performance records to CSV\n"
        << "  -h, --help          Show this help\n\n"
        << "Autotune:\n"
        << "  --autotune forces one fresh bounded CPU + GPU calibration pass. Progress is\n"
        << "  printed, winners are validated/cached, and later normal starts load them\n"
        << "  immediately without repeating the bounded calibration.\n\n"
        << "CPU tuning modes:\n"
        << "  off      no CPU benchmark; mine immediately with configured/default settings\n"
        << "  simple   quick production tuning\n"
        << "  default  balanced production tuning\n"
        << "  full     exhaustive production/rotation tuning where supported\n\n"
        << "GPU tuning modes:\n"
        << "  off      never request GPU benchmarking; use saved/safe production settings\n"
        << "  auto     default; load saved tuning and offer first-run calibration if absent\n"
        << "  full     force fresh bounded calibration plus deep production retuning\n"
        << "           as real GhostRider rotations are encountered\n\n"
        << "Developer GPU tuning environment variables still override normal operator use:\n"
        << "  YERBAS_GPU_AUTOTUNE=1          Force bounded scratchpad/variant calibration\n"
        << "  YERBAS_CUDA_RETUNE=1           Ignore exact CUDA selector/geometry caches\n"
        << "                                  and deep-retune production rotations\n"
        << "  YERBAS_GPU_VARIANT_AUTOTUNE=1  Force CN-variant batch calibration\n\n"
        << "CPU autotune environment:\n"
        << "  YERBAS_CPU_RETUNE=1            Ignore cached CPU tuning and benchmark again\n"
        << "  YERBAS_CPU_DISABLE_AUTOTUNE=1  Force direct/no-tune CPU startup\n"
        << "  YERBAS_DIAGNOSTICS=1           Show individual autotune benchmark results\n";
}

} // namespace yerbas
