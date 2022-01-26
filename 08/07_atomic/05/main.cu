#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void parallel_filter(int *sum, int *res, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        if (arr[i] >= 2) {
            int loc = atomicAdd(&sum[0], 1);
            res[loc] = arr[i];
        }
    }
}

int main() {
    int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(1);
    std::vector<int, CudaAllocator<int>> res(n);

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_filter);
    parallel_filter<<<n / 4096, 512>>>(sum.data(), res.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_filter);

    for (int i = 0; i < sum[0]; i++) {
        if (res[i] < 2) {
            printf("Wrong At %d\n", i);
            return -1;
        }
    }

    printf("All Correct!\n");
    return 0;
}
