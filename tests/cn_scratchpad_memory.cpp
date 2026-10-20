#include <sys/mman.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <cstring>
struct Unmap {std::uintptr_t address;std::size_t size;};
static std::vector<Unmap> released;
static std::uintptr_t fake_base=0x40001000U;
static bool real_mode=false,fail_mapping=false;
static unsigned fail_unmap=0;
static std::size_t requested=0;
static void* test_mmap(void* address,std::size_t size,int prot,int flags,int fd,off_t offset){
 if(real_mode)return ::mmap(address,size,prot,flags,fd,offset);
 requested=size;return fail_mapping?MAP_FAILED:reinterpret_cast<void*>(fake_base);
}
static int test_munmap(void* address,std::size_t size){
 if(real_mode)return ::munmap(address,size);
 released.push_back({reinterpret_cast<std::uintptr_t>(address),size});
 return fail_unmap==released.size()?-1:0;
}
#define mmap test_mmap
#define munmap test_munmap
#include "cpu/cn_scratchpad_memory.h"
#undef mmap
#undef munmap
static void require(bool ok){if(!ok)throw std::runtime_error("Scratchpad mapping regression");}
int main(){try{
 const std::size_t size=2097152U;
 for(std::size_t offset=0;offset<size;offset+=4096){
  fake_base=0x40000000U+offset;released.clear();requested=0;
  auto* ptr=yerbas_cn_map_aligned_scratchpad(size);
  auto aligned=reinterpret_cast<std::uintptr_t>(ptr);
  require(ptr!=MAP_FAILED&&aligned%size==0&&requested==2*size);
  const auto prefix=aligned-fake_base;const auto suffix=size-prefix;
  require(released.size()==(prefix?1U:0U)+(suffix?1U:0U));
  std::size_t index=0;
  if(prefix){require(released[index].address==fake_base&&released[index].size==prefix);++index;}
  if(suffix)require(released[index].address==aligned+size&&released[index].size==suffix);
 }
 released.clear();fail_mapping=true;require(yerbas_cn_map_aligned_scratchpad(size)==MAP_FAILED&&released.empty());fail_mapping=false;
 fake_base=0x40001000U;
 for(unsigned failure:{1U,2U}){released.clear();fail_unmap=failure;require(yerbas_cn_map_aligned_scratchpad(size)==MAP_FAILED);require(released.back().address==fake_base&&released.back().size==2*size);}fail_unmap=0;
 for(auto invalid:{std::size_t(0),std::size_t(3),std::numeric_limits<std::size_t>::max()}){requested=0;require(yerbas_cn_map_aligned_scratchpad(invalid)==MAP_FAILED&&requested==0);}
 real_mode=true;
 for(unsigned i=0;i<32;++i){auto* ptr=yerbas_cn_map_aligned_scratchpad(size);require(ptr!=MAP_FAILED&&reinterpret_cast<std::uintptr_t>(ptr)%size==0);auto* bytes=static_cast<unsigned char*>(ptr);bytes[0]=0xa5;bytes[size-1]=0x5a;require(bytes[0]==0xa5&&bytes[size-1]==0x5a);require(::munmap(ptr,size)==0);}
 std::cout<<"Scratchpad alignment PASS: all base-page offsets, trim/map failures, invalid sizes, real mapping access and release\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
