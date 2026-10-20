#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace yerbas::console::chart {
struct Series {
    const std::vector<double>* history;
    std::string foreground;
    std::string background;
};
inline double sample(const std::vector<double>& values, std::size_t x,
                     std::size_t width, std::size_t limit) {
    const auto count = std::min(limit, values.size());
    if (!count) return 0.0;
    const double p = width <= 1 ? 0.0 : double(x) * double(count - 1) / double(width - 1);
    const auto i = static_cast<std::size_t>(p);
    const auto j = std::min(i + 1, count - 1);
    const auto start = values.size() - count;
    return std::max(0.0, values[start + i] * (1.0 - (p - i)) + values[start + j] * (p - i));
}
// A terminal cell can show two colors through foreground/background blocks.
// At sub-cell boundaries choose the dominant visible band; numeric rates retain
// full precision even where a contribution is below terminal resolution.
inline std::vector<std::string> render(const std::vector<Series>& sources,
    std::size_t width, std::size_t limit, double high, int worker = -1, int quantum = 1, int height = 3) {
    height = std::clamp(height, 1, 12);
    const int levels = height * 8;
    std::vector<std::string> rows(height);
    static constexpr const char* glyphs[] = {" ","▁","▂","▃","▄","▅","▆","▇","█"};
    const int reference = static_cast<int>(sources.size());
    const auto fg = [&](int i) {return i == reference ? std::string("\x1b[38;5;237m") : sources[i].foreground;};
    const auto bg = [&](int i) {return i == reference ? std::string("\x1b[48;5;237m") : sources[i].background;};
    const auto level = [&](double v) {
        return std::clamp(static_cast<int>(std::lround(v / std::max(1.0, high) * double(levels) / quantum)) * quantum, 0, levels);
    };
    for (std::size_t x=0; x<width; ++x) {
        std::vector<int> owners(levels, -1);
        std::vector<double> rates; double total=0;
        for (const auto& s : sources) {rates.push_back(sample(*s.history,x,width,limit));total+=rates.back();}
        if (worker < 0) {
            double cumulative=0; int begin=0;
            for (std::size_t i=0;i<rates.size();++i) {
                cumulative+=rates[i]; const int end=level(cumulative);
                std::fill(owners.begin()+begin,owners.begin()+end,static_cast<int>(i));begin=end;
            }
        } else {
            std::fill(owners.begin(),owners.begin()+level(total),reference);
            if (static_cast<std::size_t>(worker)<rates.size())
                std::fill(owners.begin(),owners.begin()+level(rates[worker]),worker);
        }
        for (int row=0;row<height;++row) {
            const int base=(height-1-row)*8; int filled=0;
            while (filled<8 && owners[base+filled]>=0) ++filled;
            if (!filled) {rows[row]+=' ';continue;}
            const int lower=owners[base]; int boundary=1;
            while(boundary<filled && owners[base+boundary]==lower) ++boundary;
            rows[row]+="\x1b[0m";
            if (filled==8 && boundary<8 && quantum<8) {
                // Keep the two largest bands in a crowded cell. The old first/last
                // rule could discard the middle worker even when it dominated.
                std::vector<std::pair<int,int>> bands;
                for (int k=0;k<8;) {
                    int j=k+1;while(j<8 && owners[base+j]==owners[base+k])++j;
                    bands.push_back({owners[base+k],j-k});k=j;
                }
                std::vector<int> picks(bands.size());
                for(std::size_t i=0;i<picks.size();++i)picks[i]=static_cast<int>(i);
                std::stable_sort(picks.begin(),picks.end(),[&](int a,int b){return bands[a].second>bands[b].second;});
                int a=std::min(picks[0],picks[1]),b=std::max(picks[0],picks[1]);
                int split=std::clamp(static_cast<int>(std::lround(8.0*bands[a].second/(bands[a].second+bands[b].second))),1,7);
                rows[row]+=fg(bands[a].first)+bg(bands[b].first)+glyphs[split];
            } else if (worker >= 0 && lower == worker && boundary < filled) {
                // A partially filled cell cannot carry two colors without
                // painting above the total. Preserve the measured worker fill.
                rows[row]+=fg(worker)+(quantum==8?"#":glyphs[boundary]);
            } else {
                int chosen=lower, largest=0;
                for(int k=0;k<filled;++k) {
                    int n=0;for(int j=0;j<filled;++j)if(owners[base+j]==owners[base+k])++n;
                    if(n>largest){largest=n;chosen=owners[base+k];}
                }
                rows[row]+=fg(chosen)+(quantum==8?"#":glyphs[filled]);
            }
            rows[row]+="\x1b[0m";
        }
    }
    return rows;
}
} // namespace yerbas::console::chart
