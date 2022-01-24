#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Block %d of %d, Thread %d of %d\n",
           blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

int main() {
    kernel<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
