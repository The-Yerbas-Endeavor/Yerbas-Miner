#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

int yerbas_cuda_device_count()
{
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);

    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return 0;
    }

    check_cuda(status, "cudaGetDeviceCount failed");
    return count;
}

void yerbas_cuda_print_devices()
{
    const int count = yerbas_cuda_device_count();

    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, device),
                   "cudaGetDeviceProperties failed");

        const double memory_gib =
            static_cast<double>(props.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

        std::cout << "GPU " << device << ": " << props.name
                  << " | CC " << props.major << '.' << props.minor
                  << " | " << memory_gib << " GiB\n";
    }
}
