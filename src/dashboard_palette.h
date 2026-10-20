#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
namespace yerbas::console::chart {
struct WorkerColor { std::string foreground,background; };
class WorkerPalette {
    std::vector<unsigned> colors_;
    void add(unsigned color){if(std::find(colors_.begin(),colors_.end(),color)==colors_.end())colors_.push_back(color);}
public:
    explicit WorkerPalette(const std::string& theme="blue",const std::string& custom=""){
        std::vector<unsigned> requested;
        if(!custom.empty()){
            std::istringstream in(custom);std::string token;bool valid=custom.back()!=',';
            while(std::getline(in,token,',')){
                if(token.empty()||token.find_first_not_of("0123456789 ")!=std::string::npos){valid=false;break;}
                std::istringstream number(token);unsigned value=256;std::string extra;
                if(!(number>>value)||value>255||(number>>extra)){valid=false;break;}
                requested.push_back(value);
            }
            if(!valid)requested.clear();
        }
        for(auto color:requested)add(color);
        if(theme=="green")for(unsigned c:{29,35,120,43,78,114,36,72,150,28,34,83})add(c);
        else if(theme=="ocean")for(unsigned c:{35,37,153,69,30,117,61,80,111,45,75,159})add(c);
        else if(theme=="neutral")for(unsigned c:{242,247,253,244,250,255,240,245,248,252,238,254})add(c);
        else for(unsigned c:{25,38,153,69,30,117,61,80,111,45,75,159})add(c);
        // Extend the preferred blue/green family before adding other accents.
        for(unsigned c:{24,26,27,31,32,33,37,39,44,51,67,68,74,79,81,87,110,116,152,158,195,123,23,60,66,73,109,115,151,194})add(c);
        // Every terminal palette index is available; no device-count limit.
        for(unsigned c=16;c<256;++c)add(c);
        for(unsigned c=0;c<16;++c)add(c);
    }
    static WorkerPalette from_environment(){
        const char* theme=std::getenv("YERBAS_CHART_PALETTE");
        const char* custom=std::getenv("YERBAS_CHART_COLORS");
        return WorkerPalette(theme?theme:"blue",custom?custom:"");
    }
    WorkerColor worker(std::size_t slot)const{
        const auto value=std::to_string(colors_[slot%colors_.size()]);
        return {"\x1b[38;5;"+value+"m","\x1b[48;5;"+value+"m"};
    }
    WorkerColor cpu()const{return worker(0);}
    WorkerColor gpu(int device_id)const{return worker(device_id<0?1:static_cast<std::size_t>(device_id)+1);}
};
}
