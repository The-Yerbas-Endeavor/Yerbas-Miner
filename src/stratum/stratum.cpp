#include "stratum/stratum.h"

#include "crypto/sha256.h"
#include "ghostrider/ghostrider.h"

#include <nlohmann/json.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

namespace yerbas::stratum {
namespace {

using boost::multiprecision::cpp_int;
// GhostRider miners use a 65536 target factor. Treating pool difficulty as
// raw Bitcoin diff makes the local share target 65536x too hard and prevents
// normal vardiff shares from ever being submitted at realistic hash rates.
constexpr double kGhostRiderTargetFactor = 65536.0;
constexpr std::uint64_t kGhostRiderTargetFactorInt = 65536ULL;
constexpr double kStratumDiffOneHashes = 4294967296.0 / kGhostRiderTargetFactor;
constexpr std::uint64_t kNonceSpace = 0x100000000ULL;
constexpr std::uint32_t kHybridCpuStart = 0x80000000U;

struct CpuCandidate {
    std::uint32_t nonce{0};
    std::string extranonce2;
};

void close_socket(SocketHandle socket)
{
    if (socket == kInvalidSocket) return;
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}

std::string socket_error_string()
{
#ifdef _WIN32
    return "Winsock error " + std::to_string(WSAGetLastError());
#else
    return std::strerror(errno);
#endif
}

bool send_all(SocketHandle socket, const std::string& data)
{
    std::size_t sent = 0;
    while (sent < data.size()) {
#ifdef _WIN32
        const int n = send(socket, data.data() + sent,
                           static_cast<int>(data.size() - sent), 0);
#else
        const ssize_t n = send(socket, data.data() + sent, data.size() - sent, 0);
#endif
        if (n <= 0) return false;
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

SocketHandle connect_tcp(const Endpoint& endpoint)
{
#ifdef _WIN32
    WSADATA data{};
    const int startup = WSAStartup(MAKEWORD(2, 2), &data);
    if (startup != 0) throw std::runtime_error("WSAStartup failed: " + std::to_string(startup));
#endif

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* result = nullptr;
    const std::string port = std::to_string(endpoint.port);
    const int rc = getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &result);
    if (rc != 0) {
#ifdef _WIN32
        throw std::runtime_error("DNS lookup failed for " + endpoint.host + ": " + std::to_string(rc));
#else
        throw std::runtime_error("DNS lookup failed for " + endpoint.host + ": " + gai_strerror(rc));
#endif
    }

    SocketHandle connected = kInvalidSocket;
    for (addrinfo* ai = result; ai != nullptr; ai = ai->ai_next) {
        SocketHandle s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == kInvalidSocket) continue;
        if (connect(s, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0) {
            connected = s;
            break;
        }
        close_socket(s);
    }
    freeaddrinfo(result);

    if (connected == kInvalidSocket) {
        throw std::runtime_error("Could not connect to " + endpoint.host + ':' +
                                 std::to_string(endpoint.port) + " (" + socket_error_string() + ')');
    }
    return connected;
}

bool socket_readable(SocketHandle socket, int milliseconds)
{
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(socket, &readfds);
    timeval tv{};
    tv.tv_sec = milliseconds / 1000;
    tv.tv_usec = (milliseconds % 1000) * 1000;
#ifdef _WIN32
    const int result = select(0, &readfds, nullptr, nullptr, &tv);
#else
    const int result = select(socket + 1, &readfds, nullptr, nullptr, &tv);
#endif
    return result > 0 && FD_ISSET(socket, &readfds);
}

std::string json_line(const nlohmann::json& message) { return message.dump() + "\n"; }

int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

std::vector<std::uint8_t> hex_to_bytes(const std::string& hex)
{
    if ((hex.size() & 1U) != 0) throw std::runtime_error("Odd-length hex field in Stratum job");
    std::vector<std::uint8_t> out(hex.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const int hi = hex_value(hex[i * 2]);
        const int lo = hex_value(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) throw std::runtime_error("Invalid hex field in Stratum job");
        out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return out;
}

std::string hex_fixed(std::uint64_t value, std::size_t bytes)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(static_cast<int>(bytes * 2)) << value;
    std::string out = ss.str();
    if (out.size() > bytes * 2) out = out.substr(out.size() - bytes * 2);
    return out;
}

std::string nonce_hex(std::uint32_t nonce)
{
    // YERB-Pool parses the submitted nonce hex as an integer and serializes
    // that value little-endian into the block header. Submit the host-order
    // numeric value here so the pool reconstructs the same nonce bytes we hash.
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << nonce;
    return ss.str();
}

void write_nonce(std::array<std::uint8_t, 80>& header, std::uint32_t nonce)
{
    header[76] = static_cast<std::uint8_t>(nonce);
    header[77] = static_cast<std::uint8_t>(nonce >> 8);
    header[78] = static_cast<std::uint8_t>(nonce >> 16);
    header[79] = static_cast<std::uint8_t>(nonce >> 24);
}

std::array<std::uint8_t, 32> merkle_root(const MiningJob& job,
                                         const std::string& extranonce1,
                                         const std::string& extranonce2)
{
    std::vector<std::uint8_t> coinbase = hex_to_bytes(job.coinb1 + extranonce1 + extranonce2 + job.coinb2);
    auto hash = crypto::double_sha256(coinbase);
    for (const auto& branch_hex : job.merkle_branch) {
        const auto branch = hex_to_bytes(branch_hex);
        if (branch.size() != 32) throw std::runtime_error("Invalid merkle branch length");
        std::vector<std::uint8_t> pair;
        pair.reserve(64);
        pair.insert(pair.end(), hash.begin(), hash.end());
        pair.insert(pair.end(), branch.begin(), branch.end());
        hash = crypto::double_sha256(pair);
    }
    return hash;
}

std::vector<std::uint8_t> stratum_prevhash_bytes(const std::string& hex)
{
    auto bytes = hex_to_bytes(hex);
    if (bytes.size() != 32) throw std::runtime_error("Stratum prevhash must be 32 bytes");
    for (std::size_t i = 0; i < 32; i += 4) {
        std::reverse(bytes.begin() + static_cast<std::ptrdiff_t>(i),
                     bytes.begin() + static_cast<std::ptrdiff_t>(i + 4));
    }
    return bytes;
}

std::vector<std::uint8_t> reversed_4byte_field(const std::string& hex)
{
    auto bytes = hex_to_bytes(hex);
    if (bytes.size() != 4) throw std::runtime_error("Expected 4-byte Stratum header field");
    std::reverse(bytes.begin(), bytes.end());
    return bytes;
}

bool hash_meets_target(const ghostrider::Hash256& hash,
                       const std::array<std::uint8_t, 32>& target_le)
{
    for (int i = 31; i >= 0; --i) {
        const auto index = static_cast<std::size_t>(i);
        if (hash[index] < target_le[index]) return true;
        if (hash[index] > target_le[index]) return false;
    }
    return true;
}

cpp_int parse_hex_int(const std::string& hex)
{
    cpp_int value = 0;
    for (char c : hex) {
        const int v = hex_value(c);
        if (v < 0) throw std::runtime_error("Invalid target hex");
        value <<= 4;
        value += v;
    }
    return value;
}

std::string format_rate(double hps)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    if (hps >= 1000000.0) ss << hps / 1000000.0 << " MH/s";
    else if (hps >= 1000.0) ss << hps / 1000.0 << " kH/s";
    else ss << hps << " H/s";
    return ss.str();
}

std::string format_duration(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0) return "n/a";
    const auto total = static_cast<std::uint64_t>(seconds + 0.5);
    const auto days = total / 86400;
    const auto hours = (total % 86400) / 3600;
    const auto minutes = (total % 3600) / 60;
    const auto secs = total % 60;
    std::ostringstream ss;
    if (days) ss << days << 'd' << ' ';
    if (days || hours) ss << hours << 'h' << ' ';
    if (days || hours || minutes) ss << minutes << 'm' << ' ';
    ss << secs << 's';
    return ss.str();
}

} // namespace

Endpoint parse_endpoint(const std::string& url)
{
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) throw std::runtime_error("Pool URL must include a scheme, e.g. stratum+tcp://host:port");
    Endpoint endpoint;
    endpoint.scheme = url.substr(0, scheme_end);
    const std::string authority = url.substr(scheme_end + 3);
    const auto colon = authority.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= authority.size()) throw std::runtime_error("Pool URL must include host and port");
    endpoint.host = authority.substr(0, colon);
    const unsigned long port = std::stoul(authority.substr(colon + 1));
    if (port == 0 || port > 65535) throw std::runtime_error("Pool port is out of range");
    endpoint.port = static_cast<unsigned short>(port);
    if (endpoint.scheme != "stratum+tcp" && endpoint.scheme != "stratum") throw std::runtime_error("Unsupported pool scheme: " + endpoint.scheme);
    return endpoint;
}

Client::Client(const AppConfig& config) : config_(config)
{
    if (!config_.pool.url.empty()) endpoint_ = parse_endpoint(config_.pool.url);
#ifdef YERBAS_HAS_CUDA
    if (config_.gpu.enabled) initialize_gpu_engines();
#endif
}

void Client::print_connection_plan() const
{
    if (config_.pool.url.empty()) { std::cout << "Pool: not configured\n"; return; }
    std::cout << "Pool: " << endpoint_.host << ':' << endpoint_.port << '\n';
    std::cout << "Worker: " << config_.miner.worker << '\n';
    std::cout << "User: " << (config_.pool.user.empty() ? "not configured" : config_.pool.user) << '\n';
#ifdef YERBAS_HAS_CUDA
    if (config_.miner.cpu_enabled && config_.gpu.enabled && !gpu_workers_.empty())
        std::cout << "Stratum transport: TCP + hybrid CPU/CUDA GhostRider scheduler\n";
    else if (config_.gpu.enabled && !gpu_workers_.empty())
        std::cout << "Stratum transport: TCP + CUDA GhostRider batch scheduler\n";
    else
        std::cout << "Stratum transport: TCP + CPU GhostRider scheduler\n";
#else
    std::cout << "Stratum transport: TCP + CPU GhostRider scheduler\n";
#endif
}

bool Client::ready() const noexcept { return !config_.pool.url.empty() && !config_.pool.user.empty(); }

std::string Client::login_user() const
{
    if (config_.miner.worker.empty()) return config_.pool.user;
    return config_.pool.user + "." + config_.miner.worker;
}

int Client::run(std::atomic_bool& stop_requested)
{
    if (!ready()) { std::cerr << "Pool configuration is incomplete. Set pool.url and pool.user.\n"; return 2; }
    mining_started_ = std::chrono::steady_clock::now();
    last_report_ = mining_started_;
    hashes_at_last_report_ = hashes_done_;
    cpu_hashes_at_last_report_ = cpu_hashes_done_;

#ifdef YERBAS_HAS_CUDA
    if (config_.gpu.enabled && !gpu_workers_.empty() && !gpu_pipeline_ready_) {
        std::cout << "[GPU] Devices detected, but full GhostRider CUDA pipeline is not ready yet.\n";
        if (config_.miner.cpu_enabled) {
            std::cout << "[hybrid] CPU workers will mine; GPU workers remain idle until validated CUDA stages are complete.\n";
        } else {
            std::cerr << "No usable mining backend: CPU is disabled and CUDA pipeline is incomplete.\n";
            return 3;
        }
    }
#endif

    if (!config_.miner.cpu_enabled) {
#ifdef YERBAS_HAS_CUDA
        if (!config_.gpu.enabled || gpu_workers_.empty() || !gpu_pipeline_ready_) return 3;
#else
        return 3;
#endif
    }

    std::cout << "Starting Stratum miner. Press Ctrl+C to stop.\n";
    while (!stop_requested.load()) {
        try { if (run_session(stop_requested)) break; }
        catch (const std::exception& e) { std::cerr << "[stratum] " << e.what() << '\n'; }
        if (!stop_requested.load()) {
            std::cout << "[stratum] Reconnecting in 5 seconds...\n";
            for (int i = 0; i < 50 && !stop_requested.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    report_stats(true);
#ifdef _WIN32
    WSACleanup();
#endif
    std::cout << "Stratum miner stopped.\n";
    return 0;
}

bool Client::run_session(std::atomic_bool& stop_requested)
{
    subscribed_ = false;
    authorized_ = false;
    job_.valid = false;
#ifdef YERBAS_HAS_CUDA
    gpu_job_loaded_ = false;
#endif

    std::cout << "[stratum] Connecting to " << endpoint_.host << ':' << endpoint_.port << "...\n";
    SocketHandle socket_handle = connect_tcp(endpoint_);
    std::cout << "[stratum] Connected\n";

    const nlohmann::json subscribe = {{"id",1},{"method","mining.subscribe"},{"params",nlohmann::json::array({"Yerbas-Miner/0.5.2"})}};
    const nlohmann::json authorize = {{"id",2},{"method","mining.authorize"},{"params",nlohmann::json::array({login_user(),config_.pool.password})}};
    if (!send_all(socket_handle, json_line(subscribe)) || !send_all(socket_handle, json_line(authorize))) {
        close_socket(socket_handle);
        throw std::runtime_error("Failed to send Stratum subscribe/authorize requests");
    }

    std::string pending;
    char buffer[8192];
    while (!stop_requested.load()) {
        if (socket_readable(socket_handle, job_.valid ? 0 : 250)) {
#ifdef _WIN32
            const int n = recv(socket_handle, buffer, static_cast<int>(sizeof(buffer)), 0);
#else
            const ssize_t n = recv(socket_handle, buffer, sizeof(buffer), 0);
#endif
            if (n <= 0) { close_socket(socket_handle); std::cout << "[stratum] Connection closed by pool\n"; return false; }
            pending.append(buffer, static_cast<std::size_t>(n));
            for (;;) {
                const auto newline = pending.find('\n');
                if (newline == std::string::npos) break;
                std::string line = pending.substr(0, newline);
                pending.erase(0, newline + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) handle_message(line);
            }
        }

        if (authorized_ && job_.valid && target_ready_) {
#ifdef YERBAS_HAS_CUDA
            const bool usable_gpu = config_.gpu.enabled && !gpu_workers_.empty() && gpu_pipeline_ready_;
            if (usable_gpu && !gpu_job_loaded_) upload_gpu_job();
            if (usable_gpu && config_.miner.cpu_enabled && config_.miner.hybrid) {
                if (!mine_hybrid_round(static_cast<std::intptr_t>(socket_handle))) {
                    close_socket(socket_handle); return false;
                }
            } else if (usable_gpu) {
                if (!mine_gpu_batch(static_cast<std::intptr_t>(socket_handle))) {
                    close_socket(socket_handle); return false;
                }
            } else if (config_.miner.cpu_enabled) {
                if (!mine_cpu_batch(static_cast<std::intptr_t>(socket_handle))) {
                    close_socket(socket_handle); return false;
                }
            }
#else
            if (config_.miner.cpu_enabled && !mine_cpu_batch(static_cast<std::intptr_t>(socket_handle))) {
                close_socket(socket_handle); return false;
            }
#endif
        } else if (!socket_readable(socket_handle, 0)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        report_stats(false);
    }

    close_socket(socket_handle);
    return true;
}

void Client::handle_message(const std::string& line)
{
    nlohmann::json message;
    try { message = nlohmann::json::parse(line); }
    catch (const std::exception&) { std::cerr << "[stratum] Non-JSON message: " << line << '\n'; return; }

    if (message.contains("id") && !message["id"].is_null()) {
        const int id = message["id"].is_number_integer() ? message["id"].get<int>() : -1;
        const auto error = message.value("error", nlohmann::json(nullptr));
        if (id == 1) {
            if (error.is_null()) {
                subscribed_ = true;
                const auto result = message.value("result", nlohmann::json::array());
                if (result.is_array() && result.size() >= 3) {
                    if (result[1].is_string()) extranonce1_ = result[1].get<std::string>();
                    if (result[2].is_number_unsigned() || result[2].is_number_integer()) extranonce2_size_ = result[2].get<std::size_t>();
                }
                std::cout << "[stratum] Subscription accepted; extranonce1=" << extranonce1_ << " extranonce2_size=" << extranonce2_size_ << '\n';
            } else std::cerr << "[stratum] Subscription rejected: " << message.dump() << '\n';
        } else if (id == 2) {
            const bool ok = message.contains("result") && message["result"].is_boolean() && message["result"].get<bool>();
            if (ok && error.is_null()) { authorized_ = true; std::cout << "[stratum] Authorization accepted as " << login_user() << '\n'; }
            else std::cerr << "[stratum] Authorization rejected: " << message.dump() << '\n';
        } else if (id >= 1000) {
            bool result_ok = false;
            if (message.contains("result")) {
                const auto& result = message["result"];
                result_ok = result.is_boolean() ? result.get<bool>() : !result.is_null();
            }
            const bool accepted = error.is_null() && result_ok;
            if (accepted) {
                ++shares_accepted_;
                std::cout << "[share] ACCEPTED | accepted=" << shares_accepted_ << " rejected=" << shares_rejected_ << '\n';
            } else {
                ++shares_rejected_;
                std::cout << "[share] REJECTED | accepted=" << shares_accepted_ << " rejected=" << shares_rejected_
                          << " | response=" << message.dump() << '\n';
            }
        }
        return;
    }

    const std::string method = message.value("method", "");
    const auto params = message.value("params", nlohmann::json::array());
    if (method == "mining.notify") {
        if (!params.is_array() || params.size() < 9) { std::cerr << "[stratum] Unsupported mining.notify shape: " << message.dump() << '\n'; return; }
        try {
            MiningJob next;
            next.job_id = params[0].get<std::string>();
            next.prevhash = params[1].get<std::string>();
            next.coinb1 = params[2].get<std::string>();
            next.coinb2 = params[3].get<std::string>();
            for (const auto& branch : params[4]) next.merkle_branch.push_back(branch.get<std::string>());
            next.version = params[5].get<std::string>();
            next.nbits = params[6].get<std::string>();
            next.ntime = params[7].get<std::string>();
            next.clean_jobs = params[8].get<bool>();
            next.valid = true;
            job_ = std::move(next);
#ifdef YERBAS_HAS_CUDA
            nonce_ = (gpu_pipeline_ready_ && config_.gpu.enabled && !gpu_workers_.empty()) ? kHybridCpuStart : 0U;
            gpu_job_loaded_ = false;
#else
            nonce_ = 0U;
#endif
            if (job_.clean_jobs) ++extranonce2_counter_;
            ++received_jobs_;
            std::cout << "[stratum] New job #" << received_jobs_ << " id=" << job_.job_id
                      << " branches=" << job_.merkle_branch.size() << " clean=" << (job_.clean_jobs ? "yes" : "no") << '\n';
        } catch (const std::exception& e) {
            job_.valid = false;
            std::cerr << "[stratum] Failed to decode mining.notify: " << e.what() << '\n';
        }
        return;
    }
    if (method == "mining.set_difficulty") {
        if (params.is_array() && !params.empty() && params[0].is_number()) set_difficulty(params[0].get<double>());
        return;
    }
    if (method == "mining.set_target") {
        if (params.is_array() && !params.empty() && params[0].is_string()) set_target_hex(params[0].get<std::string>());
        return;
    }
    std::cout << "[stratum] Message: " << message.dump() << '\n';
}

void Client::set_target_hex(const std::string& target_hex)
{
    std::string hex = target_hex;
    if (hex.size() < 64) hex.insert(hex.begin(), 64 - hex.size(), '0');
    if (hex.size() != 64) throw std::runtime_error("Pool target must be 256 bits or shorter");
    const auto be = hex_to_bytes(hex);
    for (std::size_t i = 0; i < 32; ++i) target_le_[i] = be[31 - i];
    target_ready_ = true;
#ifdef YERBAS_HAS_CUDA
    gpu_job_loaded_ = false;
#endif
    std::cout << "[stratum] Explicit share target installed\n";
}

void Client::set_difficulty(double difficulty)
{
    if (!(difficulty > 0.0)) return;
    difficulty_ = difficulty;
    static const cpp_int diff1 = parse_hex_int("00000000ffff0000000000000000000000000000000000000000000000000000");
    const std::uint64_t scaled = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(difficulty * 1000000.0));
    cpp_int target = (diff1 * kGhostRiderTargetFactorInt * 1000000ULL) / scaled;
    const cpp_int max_target = (cpp_int(1) << 256) - 1;
    if (target > max_target) target = max_target;
    for (std::size_t i = 0; i < 32; ++i) {
        target_le_[i] = static_cast<std::uint8_t>(target & 0xff);
        target >>= 8;
    }
    target_ready_ = true;
#ifdef YERBAS_HAS_CUDA
    gpu_job_loaded_ = false;
#endif
    const double expected = difficulty_ * kStratumDiffOneHashes;
    std::ostringstream msg;
    msg << std::defaultfloat << std::setprecision(8)
        << "[stratum] Difficulty set to " << difficulty_
        << " | GhostRider target factor " << static_cast<std::uint64_t>(kGhostRiderTargetFactor)
        << " | average work/share ~" << std::fixed << std::setprecision(0) << expected << " hashes";
    std::cout << msg.str() << '\n';
}

bool Client::build_header(std::array<std::uint8_t, 80>& header,
                          std::string& extranonce2_hex,
                          std::uint32_t nonce) const
{
    if (!job_.valid || extranonce1_.empty()) return false;
    try {
        extranonce2_hex = hex_fixed(extranonce2_counter_, extranonce2_size_);
        const auto version = reversed_4byte_field(job_.version);
        const auto prev = stratum_prevhash_bytes(job_.prevhash);
        const auto merkle = merkle_root(job_, extranonce1_, extranonce2_hex);
        const auto ntime = reversed_4byte_field(job_.ntime);
        const auto nbits = reversed_4byte_field(job_.nbits);
        std::copy(version.begin(), version.end(), header.begin());
        std::copy(prev.begin(), prev.end(), header.begin() + 4);
        // The pool's merkle_root_from_coinbase() returns the raw SHA256d bytes,
        // and header_bytes() copies them into the serialized header unchanged.
        std::copy(merkle.begin(), merkle.end(), header.begin() + 36);
        std::copy(ntime.begin(), ntime.end(), header.begin() + 68);
        std::copy(nbits.begin(), nbits.end(), header.begin() + 72);
        write_nonce(header, nonce);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[miner] Header construction failed: " << e.what() << '\n';
        return false;
    }
}

bool Client::mine_one(std::intptr_t socket_value)
{
    std::array<std::uint8_t, 80> header{};
    std::string extranonce2;
    const std::uint32_t nonce = nonce_++;
    if (!build_header(header, extranonce2, nonce)) return true;
    const ghostrider::Work work{header.data(), header.size()};
    const auto hash = ghostrider::hash_reference(work);
    ++cpu_hashes_done_;
    ++hashes_done_;
    if (hash_meets_target(hash, target_le_)) {
        std::cout << "[CPU] candidate | job=" << job_.job_id << " nonce=" << nonce_hex(nonce) << '\n';
        return submit_share(socket_value, extranonce2, nonce);
    }
    if (nonce_ == 0) ++extranonce2_counter_;
    return true;
}

bool Client::mine_cpu_batch(std::intptr_t socket_value)
{
    if (!config_.miner.cpu_enabled) return true;

    const unsigned int threads = config_.miner.threads == 0
        ? std::max(1u, std::thread::hardware_concurrency())
        : std::max(1u, config_.miner.threads);
    const unsigned int per_thread = config_.miner.cpu_batch == 0 ? 8U : config_.miner.cpu_batch;
    const std::uint64_t total = static_cast<std::uint64_t>(threads) * per_thread;

#ifdef YERBAS_HAS_CUDA
    const bool hybrid_nonce_partition = gpu_pipeline_ready_ && config_.gpu.enabled && !gpu_workers_.empty();
#else
    const bool hybrid_nonce_partition = false;
#endif
    const std::uint64_t cpu_region_start = hybrid_nonce_partition ? kHybridCpuStart : 0ULL;
    if (static_cast<std::uint64_t>(nonce_) < cpu_region_start) nonce_ = static_cast<std::uint32_t>(cpu_region_start);

    if (static_cast<std::uint64_t>(nonce_) + total > kNonceSpace) {
        ++extranonce2_counter_;
        nonce_ = static_cast<std::uint32_t>(cpu_region_start);
    }

    std::array<std::uint8_t, 80> base_header{};
    std::string extranonce2;
    if (!build_header(base_header, extranonce2, nonce_)) return true;

    const std::uint32_t batch_start = nonce_;
    nonce_ = static_cast<std::uint32_t>(static_cast<std::uint64_t>(nonce_) + total);

    std::vector<std::future<std::vector<CpuCandidate>>> futures;
    futures.reserve(threads);
    for (unsigned int t = 0; t < threads; ++t) {
        const std::uint32_t start = batch_start + t * per_thread;
        futures.emplace_back(std::async(std::launch::async,
            [this, base_header, extranonce2, start, per_thread]() mutable {
                std::vector<CpuCandidate> found;
                for (unsigned int i = 0; i < per_thread; ++i) {
                    const std::uint32_t nonce = start + i;
                    auto header = base_header;
                    write_nonce(header, nonce);
                    const ghostrider::Work work{header.data(), header.size()};
                    const auto hash = ghostrider::hash_reference(work);
                    if (hash_meets_target(hash, target_le_)) found.push_back({nonce, extranonce2});
                }
                return found;
            }));
    }

    for (auto& future : futures) {
        const auto candidates = future.get();
        for (const auto& candidate : candidates) {
            std::cout << "[CPU] candidate | job=" << job_.job_id << " nonce=" << nonce_hex(candidate.nonce) << '\n';
            if (!submit_share(socket_value, candidate.extranonce2, candidate.nonce)) return false;
        }
    }

    cpu_hashes_done_ += total;
    hashes_done_ += total;
    return true;
}

#ifdef YERBAS_HAS_CUDA
void Client::initialize_gpu_engines()
{
    const auto available = cuda::enumerate_devices();
    std::vector<int> selected = config_.gpu.devices;
    if (selected.empty()) {
        for (const auto& info : available) selected.push_back(info.id);
    }

    for (int id : selected) {
        const auto found = std::find_if(available.begin(), available.end(), [id](const cuda::DeviceInfo& d) { return d.id == id; });
        if (found == available.end()) {
            std::cerr << "[GPU] Requested device " << id << " is not available; skipping\n";
            continue;
        }
        const std::size_t batch_size = config_.gpu.intensity > 0
            ? (static_cast<std::size_t>(1) << std::min(config_.gpu.intensity, 24))
            : 65536;
        GpuWorker worker;
        worker.device_id = id;
        worker.engine = std::make_unique<cuda::BatchEngine>(id, batch_size);
        gpu_workers_.push_back(std::move(worker));
        std::cout << "[GPU " << id << "] batch engine initialized | batch=" << batch_size << '\n';
    }

    gpu_pipeline_ready_ = !gpu_workers_.empty();
    for (const auto& worker : gpu_workers_) {
        gpu_pipeline_ready_ = gpu_pipeline_ready_ && worker.engine->hash_pipeline_ready();
    }
}

void Client::upload_gpu_job()
{
    if (gpu_workers_.empty() || !gpu_pipeline_ready_) return;
    std::array<std::uint8_t, 80> header{};
    std::string extranonce2;
    if (!build_header(header, extranonce2, 0)) throw std::runtime_error("Unable to build CUDA job header");

    cuda::JobDescriptor descriptor;
    descriptor.header = header;
    descriptor.target_le = target_le_;
    const ghostrider::Work work{descriptor.header.data(), descriptor.header.size()};
    descriptor.stages = ghostrider::stage_schedule(work);

    const std::uint64_t gpu_space = static_cast<std::uint64_t>(kHybridCpuStart);
    const std::uint64_t region_size = gpu_space / std::max<std::size_t>(1, gpu_workers_.size());
    for (std::size_t i = 0; i < gpu_workers_.size(); ++i) {
        auto& worker = gpu_workers_[i];
        worker.engine->upload_job(descriptor);
        const std::uint64_t start = i * region_size;
        const std::uint64_t end = (i + 1 == gpu_workers_.size()) ? gpu_space : (i + 1) * region_size;
        worker.region_start = static_cast<std::uint32_t>(start);
        worker.region_end = static_cast<std::uint32_t>(end - 1);
        worker.next_nonce = worker.region_start;
    }
    nonce_ = kHybridCpuStart;
    gpu_job_loaded_ = true;
    std::cout << "[hybrid] Job partitioned: " << gpu_workers_.size()
              << " GPU region(s) + CPU upper nonce region\n";
}

bool Client::mine_gpu_batch(std::intptr_t socket_value)
{
    if (!gpu_pipeline_ready_) return true;
    std::string extranonce2 = hex_fixed(extranonce2_counter_, extranonce2_size_);

    struct PendingGpu {
        GpuWorker* worker;
        std::uint32_t start;
        std::future<std::vector<cuda::Candidate>> future;
    };
    std::vector<PendingGpu> pending;
    pending.reserve(gpu_workers_.size());

    for (auto& worker : gpu_workers_) {
        const auto count = static_cast<std::uint64_t>(worker.engine->batch_size());
        if (static_cast<std::uint64_t>(worker.next_nonce) + count - 1 > worker.region_end) {
            worker.next_nonce = worker.region_start;
        }
        const std::uint32_t start = worker.next_nonce;
        worker.next_nonce = static_cast<std::uint32_t>(static_cast<std::uint64_t>(start) + count);
        auto* engine = worker.engine.get();
        pending.push_back(PendingGpu{&worker, start,
            std::async(std::launch::async, [engine, start]() { return engine->scan(start); })});
    }

    for (auto& task : pending) {
        const auto candidates = task.future.get();
        const auto count = task.worker->engine->batch_size();
        task.worker->hashes_done += count;
        hashes_done_ += count;
        for (const auto& candidate : candidates) {
            std::cout << "[GPU " << task.worker->device_id << "] candidate | job=" << job_.job_id
                      << " nonce=" << nonce_hex(candidate.nonce) << '\n';
            if (!submit_share(socket_value, extranonce2, candidate.nonce)) return false;
        }
    }
    return true;
}

bool Client::mine_hybrid_round(std::intptr_t socket_value)
{
    if (!gpu_pipeline_ready_) return mine_cpu_batch(socket_value);

    std::string extranonce2 = hex_fixed(extranonce2_counter_, extranonce2_size_);
    struct PendingGpu {
        GpuWorker* worker;
        std::uint32_t start;
        std::future<std::vector<cuda::Candidate>> future;
    };
    std::vector<PendingGpu> pending;
    pending.reserve(gpu_workers_.size());

    for (auto& worker : gpu_workers_) {
        const auto count = static_cast<std::uint64_t>(worker.engine->batch_size());
        if (static_cast<std::uint64_t>(worker.next_nonce) + count - 1 > worker.region_end) worker.next_nonce = worker.region_start;
        const std::uint32_t start = worker.next_nonce;
        worker.next_nonce = static_cast<std::uint32_t>(static_cast<std::uint64_t>(start) + count);
        auto* engine = worker.engine.get();
        pending.push_back(PendingGpu{&worker, start,
            std::async(std::launch::async, [engine, start]() { return engine->scan(start); })});
    }

    // CPU hashes its own nonce region while CUDA kernels are executing.
    if (!mine_cpu_batch(socket_value)) return false;

    for (auto& task : pending) {
        const auto candidates = task.future.get();
        const auto count = task.worker->engine->batch_size();
        task.worker->hashes_done += count;
        hashes_done_ += count;
        for (const auto& candidate : candidates) {
            std::cout << "[GPU " << task.worker->device_id << "] candidate | job=" << job_.job_id
                      << " nonce=" << nonce_hex(candidate.nonce) << '\n';
            if (!submit_share(socket_value, extranonce2, candidate.nonce)) return false;
        }
    }
    return true;
}
#endif

bool Client::submit_share(std::intptr_t socket_value,
                          const std::string& extranonce2_hex,
                          std::uint32_t nonce)
{
    const SocketHandle socket_handle = static_cast<SocketHandle>(socket_value);
    const int request_id = 1000 + static_cast<int>(shares_submitted_ % 1000000);
    const nlohmann::json submit = {
        {"id", request_id},
        {"method", "mining.submit"},
        {"params", nlohmann::json::array({login_user(), job_.job_id, extranonce2_hex, job_.ntime, nonce_hex(nonce)})}
    };
    if (!send_all(socket_handle, json_line(submit))) {
        std::cerr << "[share] Failed to send candidate share\n";
        return false;
    }
    ++shares_submitted_;
    std::cout << "[share] Submitted #" << shares_submitted_
              << " | job=" << job_.job_id
              << " extranonce2=" << extranonce2_hex
              << " ntime=" << job_.ntime
              << " nonce=" << nonce_hex(nonce) << '\n';
    return true;
}

void Client::report_stats(bool force)
{
    const auto now = std::chrono::steady_clock::now();
    if (mining_started_.time_since_epoch().count() == 0) return;
    const double since_report = std::chrono::duration<double>(now - last_report_).count();
    if (!force && since_report < 5.0) return;

    const std::uint64_t delta_hashes = hashes_done_ - hashes_at_last_report_;
    const std::uint64_t cpu_delta = cpu_hashes_done_ - cpu_hashes_at_last_report_;
    const double total_hps = since_report > 0.0 ? static_cast<double>(delta_hashes) / since_report : 0.0;
    const double cpu_hps = since_report > 0.0 ? static_cast<double>(cpu_delta) / since_report : 0.0;
    const double uptime = std::chrono::duration<double>(now - mining_started_).count();
    const double average_hps = uptime > 0.0 ? static_cast<double>(hashes_done_) / uptime : 0.0;

    if (config_.miner.cpu_enabled) {
        std::cout << "[CPU] " << format_rate(cpu_hps)
                  << " | hashes " << cpu_hashes_done_ << '\n';
    }

#ifdef YERBAS_HAS_CUDA
    if (config_.gpu.enabled && !gpu_workers_.empty()) {
        for (auto& worker : gpu_workers_) {
            const std::uint64_t gpu_delta = worker.hashes_done - worker.hashes_at_last_report;
            const double gpu_hps = since_report > 0.0 ? static_cast<double>(gpu_delta) / since_report : 0.0;
            if (gpu_pipeline_ready_) {
                std::cout << "[GPU " << worker.device_id << "] " << format_rate(gpu_hps)
                          << " | hashes " << worker.hashes_done
                          << " | batch " << worker.engine->batch_size() << '\n';
            } else {
                std::cout << "[GPU " << worker.device_id << "] idle | CUDA GhostRider pipeline incomplete\n";
            }
            worker.hashes_at_last_report = worker.hashes_done;
        }
    }
#endif

    std::cout << "[TOTAL] " << format_rate(total_hps)
              << " | avg " << format_rate(average_hps)
              << " | hashes " << hashes_done_
              << " | jobs " << received_jobs_
              << " | shares S/A/R " << shares_submitted_ << '/' << shares_accepted_ << '/' << shares_rejected_;

    if (difficulty_ > 0.0) {
        const double expected_hashes = difficulty_ * kStratumDiffOneHashes;
        const double eta = average_hps > 0.0 ? expected_hashes / average_hps : std::numeric_limits<double>::infinity();
        std::ostringstream diff_stats;
        diff_stats << std::defaultfloat << std::setprecision(8)
                   << " | diff " << difficulty_
                   << " | expected/share " << std::fixed << std::setprecision(0) << expected_hashes
                   << " | avg ETA " << format_duration(eta);
        std::cout << diff_stats.str();
    }
    std::cout << " | uptime " << format_duration(uptime) << '\n';

    last_report_ = now;
    hashes_at_last_report_ = hashes_done_;
    cpu_hashes_at_last_report_ = cpu_hashes_done_;
}

} // namespace yerbas::stratum
