#pragma once

#include "config.h"

namespace yerbas {

class Miner {
public:
    explicit Miner(AppConfig config);
    int run();

private:
    AppConfig config_;
};

} // namespace yerbas
