#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace dtf {

void allocate_device_memory(void** ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA malloc failed: ") + cudaGetErrorString(err));
    }
}

void free_device_memory(void* ptr) {
    cudaFree(ptr);
}

} // namespace dtf
