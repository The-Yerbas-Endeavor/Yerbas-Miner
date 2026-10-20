#ifdef NDEBUG
#undef NDEBUG
#endif
#include "dashboard_rate_history.h"
#include <cassert>
#include <cmath>
#include <iostream>
using yerbas::console::RateHistory;
bool near(double a,double b){return std::abs(a-b)<1e-7;}
int main(){
 RateHistory h;assert(!h.average(600));h.append(10,10,100);assert(!h.average(1800));assert(near(*h.average(600,false),100));
 h.append(40,30,300);assert(near(*h.average(600,false),250));assert(near(*h.average(20),300));assert(near(*h.average(35),9500.0/35));
 h.append(50,10,0);assert(near(*h.average(20),150));h.append(50,10,999);assert(near(*h.average(20),150));
 RateHistory cpu,gpu,total;
 for(int sec=1;sec<=90001;++sec){double a=sec<=45000?100:200,b=sec<=45000?300:500;cpu.append(sec,1,a);gpu.append(sec,1,b);total.append(sec,1,a+b);}
 assert(cpu.retained_points()<=86402);
 for(double window:{600.,1800.,3600.,21600.,86400.})assert(near(*total.average(window),*cpu.average(window)+*gpu.average(window)));
 assert(near(*cpu.average(86400),(41399.0*100+45001.0*200)/86400));
 RateHistory early;early.append(1799,1799,42);assert(!early.average(1800));early.append(1800,1,42);assert(near(*early.average(1800),42));
 RateHistory gaps;gaps.append(10,10,100);gaps.append(30,10,100);assert(near(*gaps.average(30),2000.0/30));
 std::cout<<"PASS: duration weighting, cutoff interpolation, empty/warmup windows, idle gaps,24h retention and worker totals\n";
}
