#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Block (%d,%d) of (%d,%d), Thread (%d,%d) of (%d,%d)\n",
           blockIdx.x, blockIdx.y,
           gridDim.x, gridDim.y,
           threadIdx.x, threadIdx.y,
           blockDim.x, blockDim.y);
}

int main() {
    kernel<<<dim3(2, 1, 1), dim3(3, 2, 1)>>>();
    cudaDeviceSynchronize();
    return 0;
}
