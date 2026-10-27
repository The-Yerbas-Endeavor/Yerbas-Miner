#ifdef NDEBUG
#undef NDEBUG
#endif
#include "cpu/cpu_worker_pool.h"
#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Header = std::array<std::uint8_t, 80>;

unsigned int integer(const char* text)
{
    std::size_t consumed = 0;
    const auto n = std::stoul(text, &consumed);
    if (consumed != std::string(text).size() || n == 0 || n > 256)
        throw std::invalid_argument("integer");
    return static_cast<unsigned int>(n);
}

std::map<unsigned int, Header> corpus()
{
    std::map<unsigned int, Header> headers;
    std::uint32_t state = 0x729b351dU;
    for (unsigned int attempt = 0; attempt < 10000 && headers.size() < 20; ++attempt) {
        Header h{};
        h[0] = 4;
        for (std::size_t i = 4; i < 76; ++i) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            h[i] = static_cast<std::uint8_t>(state);
        }
        unsigned int mask = 0;
        for (auto stage : yerbas::ghostrider::stage_schedule_quiet({h.data(), h.size()}))
            if (stage & yerbas::ghostrider::kCryptoNightStageFlag)
                mask |= 1U << (stage & 0x7fU);
        headers.emplace(mask, h);
    }
    if (headers.size() != 20) throw std::runtime_error("Incomplete CN corpus");
    return headers;
}

void nonce(Header& h, std::uint32_t n)
{
    for (unsigned int b = 0; b < 4; ++b) h[76 + b] = static_cast<std::uint8_t>(n >> (8 * b));
}

struct MeasurementGuard {
    MeasurementGuard() { yerbas::cpu::set_tuning_measurement_mode(true); }
    ~MeasurementGuard() { yerbas::cpu::set_tuning_measurement_mode(false); }
};
} // namespace



#include <atomic>
#include <set>
#include <thread>
#include <cassert>
bool meets(const yerbas::ghostrider::Hash256& h,const std::array<std::uint8_t,32>& target){
 for(int i=31;i>=0;--i){if(h[i]<target[i])return true;if(h[i]>target[i])return false;}return true;
}
int main(){using namespace yerbas;MeasurementGuard guard;auto headers=corpus();
 std::array<std::uint8_t,32> target;target.fill(0x7f);
 cpu::WorkerPool pool(3,1,cpu::AffinityPolicy::Unpinned);
 for(const auto& e:headers){std::set<unsigned> expected,actual;
  for(unsigned n=0;n<51;++n){auto h=e.second;nonce(h,n);if(meets(ghostrider::hash_reference({h.data(),h.size()}),target))expected.insert(n);}
  auto found=pool.run(e.second,target,0,17,nullptr);
  for(auto c:found)assert(actual.insert(c.nonce).second);
  assert(actual==expected);assert(pool.last_hashes_done()==51);
 }
 std::cout<<"PASS real worker/reference candidate parity and odd tails: all 20 combinations\n";
}
