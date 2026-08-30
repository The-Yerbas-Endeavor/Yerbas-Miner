#pragma once

#include <string>
#include <vector>

namespace yerbas::cpu {

enum class AffinityPolicy : unsigned int {
    Unpinned = 0,
    PhysicalFirst = 1,
};

struct CpuTopology {
    bool available{false};
    unsigned int logical_cpus{0};
    unsigned int physical_cores{0};
    std::vector<unsigned int> physical_first_order;
};

CpuTopology detect_cpu_topology();
const char* affinity_policy_name(AffinityPolicy policy) noexcept;
void set_runtime_affinity_policy(AffinityPolicy policy) noexcept;
AffinityPolicy runtime_affinity_policy() noexcept;
std::vector<unsigned int> affinity_cpu_order(AffinityPolicy policy, unsigned int workers);
bool pin_current_thread_to_cpu(unsigned int cpu) noexcept;

} // namespace yerbas::cpu
