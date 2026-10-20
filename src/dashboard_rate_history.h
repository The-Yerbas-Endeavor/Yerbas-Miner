#pragma once
#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
namespace yerbas::console {
// Integrate actual report durations, independently of smoothed chart samples.
class RateHistory {
    struct Point { double time; long double work; };
    std::deque<Point> points_;
    double started_{0};
    static constexpr double retention=86400.0;
public:
    void append(double end,double duration,double hps) {
        if(!std::isfinite(end)||!std::isfinite(duration)||!std::isfinite(hps)||end<0||duration<=0||hps<0)return;
        double begin=std::max(0.0,end-duration);
        if(!points_.empty() && end<points_.back().time-1e-6)points_.clear();
        if(points_.empty()){started_=begin;points_.push_back({begin,0});}
        if(end<=points_.back().time)return;
        begin=std::max(begin,points_.back().time);
        if(begin>points_.back().time+1e-6)points_.push_back({begin,points_.back().work});
        const auto work=points_.back().work+static_cast<long double>(end-begin)*hps;
        points_.push_back({end,work});
        while(points_.size()>2 && points_[1].time<=end-retention)points_.pop_front();
    }
    std::optional<double> average(double seconds,bool require_full=true) const {
        if(points_.size()<2||!std::isfinite(seconds)||seconds<=0||seconds>retention)return std::nullopt;
        const double end=points_.back().time;
        if(require_full && end-started_+1e-6<seconds)return std::nullopt;
        const double begin=std::max(started_,end-seconds);
        if(end<=begin)return std::nullopt;
        const auto it=std::lower_bound(points_.begin(),points_.end(),begin,[](const Point& p,double t){return p.time<t;});
        long double work=points_.front().work;
        if(it==points_.end())return std::nullopt;
        if(it==points_.begin()||it->time==begin)work=it->work;
        else {const auto& left=*(it-1);work=left.work+(it->work-left.work)*((begin-left.time)/(it->time-left.time));}
        return static_cast<double>((points_.back().work-work)/(end-begin));
    }
    std::size_t retained_points()const{return points_.size();}
};
}
