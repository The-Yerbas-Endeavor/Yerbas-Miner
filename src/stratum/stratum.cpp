#include "stratum/stratum.h"

#include <iostream>
#include <stdexcept>

namespace yerbas::stratum {

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
    std::cout << "Stratum transport: scaffold (socket/session protocol next)\n";
}

bool Client::ready() const noexcept
{
    return !config_.pool.url.empty() && !config_.pool.user.empty();
}

} // namespace yerbas::stratum
