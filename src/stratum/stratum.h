#pragma once

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

private:
    AppConfig config_;
    Endpoint endpoint_;
};

} // namespace yerbas::stratum
