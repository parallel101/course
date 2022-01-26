#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
    int local_sum = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        local_sum += arr[i];
    }
    atomicAdd(&sum[0], local_sum);
}

int main() {
    int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(1);

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_sum);
    parallel_sum<<<n / 4096, 512>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_sum);

    printf("result: %d\n", sum[0]);

    return 0;
}
