#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void reverse(int *in, int *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int tmp[128];
    tmp[threadIdx.x] = in[(n - 1 - i) / 128 + threadIdx.x];
    __syncthreads();
    out[i] = tmp[127 - threadIdx.x];
}

int main() {
    int n = 1<<26;
    std::vector<int, CudaAllocator<int>> in(n);
    std::vector<int, CudaAllocator<int>> out(n);

    for (int i = 0; i < n; i++) {
        in[i] = i;
    }

    TICK(reverse);
    reverse<<<n / 128, 128>>>(in.data(), out.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(reverse);

    for (int i = 0; i < n; i++) {
        if (out[i] != in[n - 1 - i]) {
            printf("Wrong At %d: %d\n", i);
            return -1;
        }
    }
    printf("All Correct!\n");

    return 0;
}
