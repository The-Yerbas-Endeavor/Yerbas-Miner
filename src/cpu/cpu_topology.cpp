#include "cpu/cpu_topology.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace yerbas::cpu {
namespace {

std::atomic_uint g_affinity_policy{static_cast<unsigned int>(AffinityPolicy::Unpinned)};

#if defined(__linux__)
bool read_uint_file(const std::string& path, unsigned int& value)
{
    std::ifstream in(path);
    return static_cast<bool>(in >> value);
}
#endif

} // namespace

CpuTopology detect_cpu_topology()
{
    CpuTopology out{};
    out.logical_cpus = std::max(1u, std::thread::hardware_concurrency());

#if defined(__linux__)
    std::map<std::pair<unsigned int, unsigned int>, std::vector<unsigned int>> cores;
    for (unsigned int cpu = 0; cpu < out.logical_cpus; ++cpu) {
        unsigned int package_id = 0;
        unsigned int core_id = 0;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        if (!read_uint_file(base + "physical_package_id", package_id) ||
            !read_uint_file(base + "core_id", core_id))
            continue;
        cores[{package_id, core_id}].push_back(cpu);
    }

    if (!cores.empty()) {
        out.available = true;
        out.physical_cores = static_cast<unsigned int>(cores.size());
        std::size_t sibling = 0;
        for (;;) {
            bool added = false;
            for (const auto& entry : cores) {
                if (sibling < entry.second.size()) {
                    out.physical_first_order.push_back(entry.second[sibling]);
                    added = true;
                }
            }
            if (!added) break;
            ++sibling;
        }
    }
#endif

    if (out.physical_first_order.empty()) {
        out.physical_first_order.reserve(out.logical_cpus);
        for (unsigned int cpu = 0; cpu < out.logical_cpus; ++cpu)
            out.physical_first_order.push_back(cpu);
        out.physical_cores = out.logical_cpus;
    }
    return out;
}

const char* affinity_policy_name(AffinityPolicy policy) noexcept
{
    return policy == AffinityPolicy::PhysicalFirst ? "physical-first" : "unpinned";
}

void set_runtime_affinity_policy(AffinityPolicy policy) noexcept
{
    g_affinity_policy.store(static_cast<unsigned int>(policy), std::memory_order_relaxed);
}

AffinityPolicy runtime_affinity_policy() noexcept
{
    const auto value = g_affinity_policy.load(std::memory_order_relaxed);
    return value == static_cast<unsigned int>(AffinityPolicy::PhysicalFirst)
        ? AffinityPolicy::PhysicalFirst
        : AffinityPolicy::Unpinned;
}

std::vector<unsigned int> affinity_cpu_order(AffinityPolicy policy, unsigned int workers)
{
    std::vector<unsigned int> out;
    if (policy != AffinityPolicy::PhysicalFirst || workers == 0) return out;
    const auto topology = detect_cpu_topology();
    if (!topology.available || topology.physical_first_order.empty()) return out;
    const auto count = std::min<std::size_t>(workers, topology.physical_first_order.size());
    out.assign(topology.physical_first_order.begin(), topology.physical_first_order.begin() + count);
    return out;
}

bool pin_current_thread_to_cpu(unsigned int cpu) noexcept
{
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (cpu >= CPU_SETSIZE) return false;
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

} // namespace yerbas::cpu
