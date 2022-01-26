#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

template <int blockSize, class T>
__global__ void parallel_sum_kernel(T *sum, T const *arr, int n) {
    __shared__ volatile int local_sum[blockSize];
    int j = threadIdx.x;
    int i = blockIdx.x;
    T temp_sum = 0;
    for (int t = i * blockSize + j; t < n; t += blockSize * gridDim.x) {
        temp_sum += arr[t];
    }
    local_sum[j] = temp_sum;
    __syncthreads();
    if constexpr (blockSize >= 1024) {
        if (j < 512)
            local_sum[j] += local_sum[j + 512];
        __syncthreads();
    }
    if constexpr (blockSize >= 512) {
        if (j < 256)
            local_sum[j] += local_sum[j + 256];
        __syncthreads();
    }
    if constexpr (blockSize >= 256) {
        if (j < 128)
            local_sum[j] += local_sum[j + 128];
        __syncthreads();
    }
    if constexpr (blockSize >= 128) {
        if (j < 64)
            local_sum[j] += local_sum[j + 64];
        __syncthreads();
    }
    if (j < 32) {
        if constexpr (blockSize >= 64)
            local_sum[j] += local_sum[j + 32];
        if constexpr (blockSize >= 32)
            local_sum[j] += local_sum[j + 16];
        if constexpr (blockSize >= 16)
            local_sum[j] += local_sum[j + 8];
        if constexpr (blockSize >= 8)
            local_sum[j] += local_sum[j + 4];
        if constexpr (blockSize >= 4)
            local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}

template <int reduceScale = 4096, int blockSize = 256, class T>
int parallel_sum(T const *arr, int n) {
    std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
    parallel_sum_kernel<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    T final_sum = 0;
    for (int i = 0; i < n / reduceScale; i++) {
        final_sum += sum[i];
    }
    return final_sum;
}

int main() {
    int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 4096);

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_sum);
    int final_sum = parallel_sum(arr.data(), n);
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);

    return 0;
}
