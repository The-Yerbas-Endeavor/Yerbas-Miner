#pragma once
#include <cmath>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
namespace yerbas::console::power {
inline bool valid(double watts,bool available){return available&&std::isfinite(watts)&&watts>0;}
inline std::optional<double> efficiency(double hps,double watts,bool available){
    if(!valid(watts,available)||!std::isfinite(hps)||hps<0)return std::nullopt;
    const double result=hps/watts;return std::isfinite(result)?std::optional<double>(result):std::nullopt;
}
inline std::optional<double> energy(double hps,double watts,bool available){
    if(!valid(watts,available)||!std::isfinite(hps)||hps<=0)return std::nullopt;
    const double result=1000.0*watts/hps;return std::isfinite(result)?std::optional<double>(result):std::nullopt;
}
inline std::string format(std::optional<double> value,const char* unit,int precision=2){
    if(!value)return "n/a";
    std::ostringstream out;out<<std::fixed<<std::setprecision(precision)<<*value<<' '<<unit;return out.str();
}
inline std::string efficiency_text(double hps,double watts,bool available){return format(efficiency(hps,watts,available),"H/s/W");}
inline std::string energy_text(double hps,double watts,bool available){return format(energy(hps,watts,available),"J/kH",1);}
struct Summary {
    double hps{0},watts{0};unsigned workers{0},missing{0};
    void add(double rate,double power,bool available){
        ++workers;
        if(!valid(power,available)||!std::isfinite(rate)||rate<0){++missing;return;}
        hps+=rate;watts+=power;
    }
    bool complete()const{return workers>0&&missing==0&&valid(watts,true);}
    std::string power_text()const{return format(complete()?std::optional<double>(watts):std::nullopt,"W",1);}
    std::string efficiency_text()const{return power::efficiency_text(hps,watts,complete());}
    std::string energy_text()const{return power::energy_text(hps,watts,complete());}
};
}
