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
int main(){using namespace yerbas;MeasurementGuard guard;auto headers=corpus();
 for(auto& e:headers){auto h=e.second;nonce(h,7);assert(ghostrider::hash_optimized({h.data(),h.size()})==ghostrider::hash_reference({h.data(),h.size()}));}
 std::cout<<"PASS reference parity: all 20 combinations\n"<<std::flush;
 std::array<std::uint8_t,32> target;target.fill(255);unsigned total_calls=0;
 for(unsigned threads:{1U,2U,3U,6U})for(unsigned lanes:{1U,2U,4U}){
  cpu::WorkerPool pool(threads,lanes,cpu::AffinityPolicy::Unpinned);std::atomic_bool cancel{false};
  auto found=pool.run(headers.at(7),target,0xfffffff0U,17,&cancel,[&]{++total_calls;});
  std::set<unsigned> actual,expected;for(auto c:found)actual.insert(c.nonce);for(unsigned n=0;n<threads*17;++n)expected.insert(0xfffffff0U+n);
  assert(actual==expected);assert(found.size()==expected.size());assert(pool.last_hashes_done()==found.size());
  cancel.store(true);assert(pool.run(headers.at(7),target,0,48,&cancel,[]{}).empty());assert(pool.last_hashes_done()==0);
 }
 assert(total_calls>0);std::cout<<"PASS serviced API: coverage/wrap/lanes/pre-cancel, callbacks="<<total_calls<<'\n'<<std::flush;
 cpu::WorkerPool pool(6,1,cpu::AffinityPolicy::Unpinned);std::atomic_bool cancel{false};unsigned calls=0;
 auto stopped=pool.run(headers.at(7),target,0,4096,&cancel,[&]{if(++calls>=2)cancel.store(true);});
 assert(calls>=2);assert(pool.last_hashes_done()==stopped.size());assert(stopped.size()<6*4096);std::set<unsigned> seen;for(auto c:stopped)assert(seen.insert(c.nonce).second);
 std::cout<<"PASS callback cancellation: actual="<<stopped.size()<<'\n'<<std::flush;
 cancel.store(false);bool caught=false;
 try{pool.run(headers.at(7),target,0,4096,&cancel,[&]{cancel.store(true);throw std::runtime_error("service-sentinel");});}catch(const std::runtime_error&e){caught=std::string(e.what())=="service-sentinel";}
 assert(caught);assert(pool.last_hashes_done()<6*4096);
 cancel.store(false);auto after=pool.run(headers.at(7),target,100,3,&cancel);assert(after.size()==18);assert(pool.last_hashes_done()==18);
 std::cout<<"PASS callback exception drains workers and pool reuse remains valid\n"<<std::flush;
}
