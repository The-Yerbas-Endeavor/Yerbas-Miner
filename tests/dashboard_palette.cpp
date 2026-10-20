#ifdef NDEBUG
#undef NDEBUG
#endif
#include "dashboard_palette.h"
#include <cassert>
#include <set>
#include <iostream>
using yerbas::console::chart::WorkerPalette;
int main(){
 WorkerPalette p;assert(p.cpu().foreground=="\x1b[38;5;25m");assert(p.gpu(0).foreground=="\x1b[38;5;38m");assert(p.gpu(1).background=="\x1b[48;5;153m");
 for(auto name:{"blue","green","ocean","neutral"}){WorkerPalette q(name);std::set<std::string> seen;for(int i=0;i<256;++i)assert(seen.insert(q.worker(i).foreground).second);assert(q.worker(256).foreground==q.cpu().foreground);}
 for(int id:{0,7,2,15,3,31})assert(p.gpu(id).foreground==p.worker(id+1).foreground);
 WorkerPalette custom("blue","200,201,202");assert(custom.cpu().foreground=="\x1b[38;5;200m");assert(custom.gpu(0).background=="\x1b[48;5;201m");assert(custom.gpu(2).foreground!=custom.gpu(1).foreground);
 for(auto bad:{"300","1,","1,,2","-1","1,no","99999999999999999999999"})assert(WorkerPalette("blue",bad).cpu().foreground==p.cpu().foreground);
 assert(WorkerPalette("unknown").cpu().foreground==p.cpu().foreground);
 std::cout<<"PASS: blue defaults, four themes,256 distinct indices, sparse IDs, custom extension and invalid-input fallback\n";
}
