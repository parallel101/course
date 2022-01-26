#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__device__ __inline__ int float_atomic_add(float *dst, float src) {
    int old = __float_as_int(*dst), expect;
    do {
        expect = old;
        old = atomicCAS((int *)dst, expect,
                    __float_as_int(__int_as_float(expect) + src));
    } while (expect != old);
    return old;
}

__global__ void parallel_sum(float *sum, float const *arr, int n) {
    float local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }
    float_atomic_add(&sum[0], local_sum);
}

int main() {
    int n = 65536;
    std::vector<float, CudaAllocator<float>> arr(n);
    std::vector<float, CudaAllocator<float>> sum(1);

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_sum);
    parallel_sum<<<n / 4096, 512>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_sum);

    printf("result: %f\n", sum[0]);

    return 0;
}
