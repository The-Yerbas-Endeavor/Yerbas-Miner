#include "stratum/stratum.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

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

void close_socket(SocketHandle socket)
{
    if (socket == kInvalidSocket) {
        return;
    }
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
        if (n <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

SocketHandle connect_tcp(const Endpoint& endpoint)
{
#ifdef _WIN32
    WSADATA data{};
    const int startup = WSAStartup(MAKEWORD(2, 2), &data);
    if (startup != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(startup));
    }
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
        throw std::runtime_error("DNS lookup failed for " + endpoint.host + ": " +
                                 std::to_string(rc));
#else
        throw std::runtime_error("DNS lookup failed for " + endpoint.host + ": " +
                                 gai_strerror(rc));
#endif
    }

    SocketHandle connected = kInvalidSocket;
    for (addrinfo* ai = result; ai != nullptr; ai = ai->ai_next) {
        SocketHandle socket_handle = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (socket_handle == kInvalidSocket) {
            continue;
        }

        if (connect(socket_handle, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0) {
            connected = socket_handle;
            break;
        }

        close_socket(socket_handle);
    }

    freeaddrinfo(result);

    if (connected == kInvalidSocket) {
        throw std::runtime_error("Could not connect to " + endpoint.host + ':' +
                                 std::to_string(endpoint.port) + " (" +
                                 socket_error_string() + ')');
    }
    return connected;
}

bool socket_readable(SocketHandle socket, int seconds)
{
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(socket, &readfds);

    timeval tv{};
    tv.tv_sec = seconds;
    tv.tv_usec = 0;

#ifdef _WIN32
    const int result = select(0, &readfds, nullptr, nullptr, &tv);
#else
    const int result = select(socket + 1, &readfds, nullptr, nullptr, &tv);
#endif
    return result > 0 && FD_ISSET(socket, &readfds);
}

std::string json_line(const nlohmann::json& message)
{
    return message.dump() + "\n";
}

} // namespace

Endpoint parse_endpoint(const std::string& url)
{
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw std::runtime_error("Pool URL must include a scheme, e.g. stratum+tcp://host:port");
    }

    Endpoint endpoint;
    endpoint.scheme = url.substr(0, scheme_end);
    const std::string authority = url.substr(scheme_end + 3);

    const auto colon = authority.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= authority.size()) {
        throw std::runtime_error("Pool URL must include host and port");
    }

    endpoint.host = authority.substr(0, colon);
    const unsigned long port = std::stoul(authority.substr(colon + 1));
    if (port == 0 || port > 65535) {
        throw std::runtime_error("Pool port is out of range");
    }
    endpoint.port = static_cast<unsigned short>(port);

    if (endpoint.scheme != "stratum+tcp" && endpoint.scheme != "stratum") {
        throw std::runtime_error("Unsupported pool scheme: " + endpoint.scheme);
    }

    return endpoint;
}

Client::Client(const AppConfig& config)
    : config_(config)
{
    if (!config_.pool.url.empty()) {
        endpoint_ = parse_endpoint(config_.pool.url);
    }
}

void Client::print_connection_plan() const
{
    if (config_.pool.url.empty()) {
        std::cout << "Pool: not configured\n";
        return;
    }

    std::cout << "Pool: " << endpoint_.host << ':' << endpoint_.port << '\n';
    std::cout << "Worker: " << config_.miner.worker << '\n';
    std::cout << "User: " << (config_.pool.user.empty() ? "not configured" : config_.pool.user) << '\n';
    std::cout << "Stratum transport: TCP enabled\n";
}

bool Client::ready() const noexcept
{
    return !config_.pool.url.empty() && !config_.pool.user.empty();
}

std::string Client::login_user() const
{
    if (config_.miner.worker.empty()) {
        return config_.pool.user;
    }
    return config_.pool.user + "." + config_.miner.worker;
}

int Client::run(std::atomic_bool& stop_requested)
{
    if (!ready()) {
        std::cerr << "Pool configuration is incomplete. Set pool.url and pool.user.\n";
        return 2;
    }

    std::cout << "Starting Stratum client. Press Ctrl+C to stop.\n";
    while (!stop_requested.load()) {
        try {
            if (run_session(stop_requested)) {
                break;
            }
        } catch (const std::exception& e) {
            std::cerr << "[stratum] " << e.what() << '\n';
        }

        if (!stop_requested.load()) {
            std::cout << "[stratum] Reconnecting in 5 seconds...\n";
            for (int i = 0; i < 50 && !stop_requested.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

#ifdef _WIN32
    WSACleanup();
#endif
    std::cout << "Stratum client stopped.\n";
    return 0;
}

bool Client::run_session(std::atomic_bool& stop_requested)
{
    subscribed_ = false;
    authorized_ = false;

    std::cout << "[stratum] Connecting to " << endpoint_.host << ':' << endpoint_.port << "...\n";
    SocketHandle socket_handle = connect_tcp(endpoint_);
    std::cout << "[stratum] Connected\n";

    const nlohmann::json subscribe = {
        {"id", 1},
        {"method", "mining.subscribe"},
        {"params", nlohmann::json::array({"Yerbas-Miner/0.3.0"})}
    };
    const nlohmann::json authorize = {
        {"id", 2},
        {"method", "mining.authorize"},
        {"params", nlohmann::json::array({login_user(), config_.pool.password})}
    };

    if (!send_all(socket_handle, json_line(subscribe)) ||
        !send_all(socket_handle, json_line(authorize))) {
        close_socket(socket_handle);
        throw std::runtime_error("Failed to send Stratum subscribe/authorize requests");
    }

    std::string pending;
    char buffer[8192];

    while (!stop_requested.load()) {
        if (!socket_readable(socket_handle, 1)) {
            continue;
        }

#ifdef _WIN32
        const int n = recv(socket_handle, buffer, static_cast<int>(sizeof(buffer)), 0);
#else
        const ssize_t n = recv(socket_handle, buffer, sizeof(buffer), 0);
#endif
        if (n <= 0) {
            close_socket(socket_handle);
            std::cout << "[stratum] Connection closed by pool\n";
            return false;
        }

        pending.append(buffer, static_cast<std::size_t>(n));
        for (;;) {
            const auto newline = pending.find('\n');
            if (newline == std::string::npos) {
                break;
            }
            std::string line = pending.substr(0, newline);
            pending.erase(0, newline + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                handle_message(line);
            }
        }
    }

    close_socket(socket_handle);
    return true;
}

void Client::handle_message(const std::string& line)
{
    nlohmann::json message;
    try {
        message = nlohmann::json::parse(line);
    } catch (const std::exception&) {
        std::cerr << "[stratum] Non-JSON message: " << line << '\n';
        return;
    }

    if (message.contains("id") && !message["id"].is_null()) {
        const int id = message["id"].is_number_integer() ? message["id"].get<int>() : -1;
        if (id == 1) {
            if (message.value("error", nlohmann::json(nullptr)).is_null()) {
                subscribed_ = true;
                std::cout << "[stratum] Subscription accepted\n";
            } else {
                std::cerr << "[stratum] Subscription rejected: " << message.dump() << '\n';
            }
        } else if (id == 2) {
            const bool ok = message.contains("result") &&
                            ((message["result"].is_boolean() && message["result"].get<bool>()) ||
                             !message["result"].is_null());
            if (ok && message.value("error", nlohmann::json(nullptr)).is_null()) {
                authorized_ = true;
                std::cout << "[stratum] Authorization accepted as " << login_user() << '\n';
            } else {
                std::cerr << "[stratum] Authorization rejected: " << message.dump() << '\n';
            }
        }
        return;
    }

    const std::string method = message.value("method", "");
    if (method == "mining.notify") {
        ++received_jobs_;
        std::cout << "[stratum] Mining job #" << received_jobs_ << " received";
        if (message.contains("params") && message["params"].is_array() && !message["params"].empty()) {
            std::cout << " (job id " << message["params"][0].dump() << ')';
        }
        std::cout << '\n';
        std::cout << "[miner] Job is queued for GhostRider header decoding; share submission remains gated until the pool job format is verified.\n";
        return;
    }

    if (method == "mining.set_difficulty") {
        std::cout << "[stratum] Difficulty update: " << message.value("params", nlohmann::json::array()).dump() << '\n';
        return;
    }

    if (method == "mining.set_target") {
        std::cout << "[stratum] Target update received\n";
        return;
    }

    std::cout << "[stratum] Message: " << message.dump() << '\n';
}

} // namespace yerbas::stratum
