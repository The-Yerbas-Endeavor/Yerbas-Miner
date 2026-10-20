#pragma once

#if defined(__linux__)
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>

// Reserve twice the scratchpad size and trim to an aligned slice. This changes
// virtual address placement only; callers still own exactly size bytes.
static inline void* yerbas_cn_map_aligned_scratchpad(size_t size)
{
    if (size == 0U || (size & (size - 1U)) != 0U || size > SIZE_MAX / 2U)
        return MAP_FAILED;
    const size_t reserved = size * 2U;
    void* mapping = mmap(NULL, reserved, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) return MAP_FAILED;

    const uintptr_t base = (uintptr_t)mapping;
    const uintptr_t aligned = (base + size - 1U) & ~((uintptr_t)size - 1U);
    const size_t prefix = (size_t)(aligned - base);
    const size_t suffix = reserved - prefix - size;
    if ((prefix != 0U && munmap(mapping, prefix) != 0) ||
        (suffix != 0U && munmap((void*)(aligned + size), suffix) != 0)) {
        munmap(mapping, reserved);
        return MAP_FAILED;
    }
    return (void*)aligned;
}
#endif
