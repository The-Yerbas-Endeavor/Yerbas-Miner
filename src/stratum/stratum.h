#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "config.h"

namespace yerbas::stratum {

struct Endpoint {
    std::string scheme;
    std::string host;
    unsigned short port{0};
};

Endpoint parse_endpoint(const std::string& url);

class Client {
public:
    explicit Client(const AppConfig& config);

    void print_connection_plan() const;
    bool ready() const noexcept;

    // Runs until stop_requested becomes true. The client reconnects after
    // network failures and performs subscribe/authorize on each connection.
    int run(std::atomic_bool& stop_requested);

private:
    bool run_session(std::atomic_bool& stop_requested);
    void handle_message(const std::string& line);
    std::string login_user() const;

    AppConfig config_;
    Endpoint endpoint_;
    std::uint64_t received_jobs_{0};
    bool subscribed_{false};
    bool authorized_{false};
};

} // namespace yerbas::stratum
