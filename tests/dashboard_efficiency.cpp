#ifdef NDEBUG
#undef NDEBUG
#endif
#include "dashboard_efficiency.h"
#include <cassert>
#include <cmath>
#include <limits>
using namespace yerbas::console::power;
int main(){
 assert(std::abs(*efficiency(600,120,true)-5.0)<1e-9);
 assert(std::abs(*energy(600,120,true)-200.0)<1e-9);
 Summary total;total.add(600,120,true);total.add(1400,280,true);
 assert(total.complete());assert(total.power_text()=="400.0 W");assert(total.efficiency_text()=="5.00 H/s/W");assert(total.energy_text()=="200.0 J/kH");
 total.add(100,0,false);assert(!total.complete());assert(total.power_text()=="n/a");
 assert(!efficiency(100,0,true));assert(!energy(0,100,true));assert(*efficiency(0,100,true)==0);
 assert(!efficiency(-1,100,true));assert(!efficiency(1,-1,true));
 assert(!efficiency(std::numeric_limits<double>::infinity(),100,true));
 assert(!efficiency(100,std::numeric_limits<double>::quiet_NaN(),true));
 Summary many;for(int i=0;i<32;++i)many.add(1000,200,true);assert(many.complete());assert(many.efficiency_text()=="5.00 H/s/W");
}
