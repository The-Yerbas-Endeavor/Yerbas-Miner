#include "cuda/gpu_cache_limits.h"
#include <iostream>
#include <stdexcept>
#include <limits>
using yerbas::gpu_cache::adapt_batches;
void require(bool ok) { if (!ok) throw std::runtime_error("GPU cache limits regression"); }
int main() { try {
 constexpr std::size_t MiB=1024U*1024U;
 const std::array<std::size_t,4> pages{{2*MiB,MiB,MiB/2,MiB/4}};
 const std::array<std::size_t,4> saved{{3584,8064,13440,13440}};
 auto current=saved;require(adapt_batches(current,pages,32768,0,31744,7936*MiB,896));
 require(current==std::array<std::size_t,4>{{3584,7168,13440,13440}});
 auto same=saved;require(adapt_batches(same,pages,32768,8192*MiB,32768,8192*MiB,896));require(same==saved);
 auto larger=saved;require(adapt_batches(larger,pages,32768,8192*MiB,65536,16384*MiB,896));require(larger==saved);
 for (auto invalid : {std::size_t(0),std::size_t(32769),std::numeric_limits<std::size_t>::max()}) {auto x=saved;x[0]=invalid;auto before=x;require(!adapt_batches(x,pages,32768,8192*MiB,31744,7936*MiB,896));require(x==before);}
 auto x=saved;x[1]=8065;require(!adapt_batches(x,pages,32768,8192*MiB,31744,7936*MiB,896));
 x=saved;require(!adapt_batches(x,pages,32768,8192*MiB,31744,0,896));require(!adapt_batches(x,pages,32768,8192*MiB,31744,7936*MiB,0));
 auto badpages=pages;badpages[0]=0;require(!adapt_batches(x,badpages,32768,8192*MiB,31744,7936*MiB,896));
 const std::array<std::size_t,6> vp{{MiB/2,MiB/2,2*MiB,MiB,MiB/4,MiB/4}};
 std::array<std::size_t,6> variants{{9856,9856,3584,5376,9856,19712}};auto old=variants;require(adapt_batches(variants,vp,32768,8192*MiB,31744,7936*MiB,896));require(variants==old);
 auto impossible=old;impossible[2]=4480;require(!adapt_batches(impossible,vp,32768,8192*MiB,31744,7936*MiB,896));
 for(std::size_t cap=1;cap<=40000;cap+=131)for(std::size_t budget=1;budget<=10000*MiB;budget+=137*MiB){auto z=saved;bool ok=adapt_batches(z,pages,32768,8192*MiB,cap,budget,896);if(ok)for(std::size_t i=0;i<z.size();++i){require(z[i]>0&&z[i]<=saved[i]&&z[i]<=cap&&z[i]<=budget/pages[i]);require(z[i]<896||z[i]%896==0);}}
 std::cout<<"GPU cache limits PASS: observed capacity regression, unchanged/increased budgets, corrupt records, variant compatibility, and capacity/budget sweep\n";return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
