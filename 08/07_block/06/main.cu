#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n / 1024; i += blockDim.x * gridDim.x) {
        int local_sum[1024];
        for (int j = 0; j < 1024; j++) {
            local_sum[j] = arr[i * 1024 + j];
        }
        for (int j = 0; j < 512; j++) {
            local_sum[j] += local_sum[j + 512];
        }
        for (int j = 0; j < 256; j++) {
            local_sum[j] += local_sum[j + 256];
        }
        for (int j = 0; j < 128; j++) {
            local_sum[j] += local_sum[j + 128];
        }
        for (int j = 0; j < 64; j++) {
            local_sum[j] += local_sum[j + 64];
        }
        for (int j = 0; j < 32; j++) {
            local_sum[j] += local_sum[j + 32];
        }
        for (int j = 0; j < 16; j++) {
            local_sum[j] += local_sum[j + 16];
        }
        for (int j = 0; j < 8; j++) {
            local_sum[j] += local_sum[j + 8];
        }
        for (int j = 0; j < 4; j++) {
            local_sum[j] += local_sum[j + 4];
        }
        for (int j = 0; j < 2; j++) {
            local_sum[j] += local_sum[j + 2];
        }
        for (int j = 0; j < 1; j++) {
            local_sum[j] += local_sum[j + 1];
        }
        sum[i] = local_sum[0];
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
    parallel_sum<<<n / 1024 / 128, 128>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());

    int final_sum = 0;
    for (int i = 0; i < n / 1024; i++) {
        final_sum += sum[i];
    }
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);

    return 0;
}
