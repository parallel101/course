#include <cuda_runtime.h>
#include "better_cuda.cuh"
using namespace cupp;

/*__host__*/ void host_func() {
}

/*__host__ __device__*/ constexpr void constexpr_func() {
}

__device__ void device_func() {
}

__host__ __device__ void host_device_func() {
}

__global__ void kernel() {
    device_func();
    host_device_func();
    constexpr_func(); // 需开启 --expt-relaxed-constexpr
    auto device_lambda = [] __device__ (int i) { // 需开启 --expt-extended-lambda
        return i * 2;
    };
    device_lambda(1);
}

int main() {
    host_func();
    host_device_func();
    constexpr_func();
    auto host_lambda = [] (int i) {
        return i * 2;
    };
    host_lambda(1);
    CudaAllocator<char, CudaAsyncDeviceArena> a;
    auto p = CudaMemPool::Builder().withLocation(cudaMemLocationTypeDevice).build();
}
