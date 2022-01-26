#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    local_sum[j] = arr[i * 1024 + j];
    __syncthreads();
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}

int main() {
    int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 1024);

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_sum);
    parallel_sum<<<n / 1024, 1024>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());

    int final_sum = 0;
    for (int i = 0; i < n / 1024; i++) {
        final_sum += sum[i];
    }
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);

    return 0;
}
