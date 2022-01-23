#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Thread %d of %d\n", threadIdx.x, blockDim.x);
}

int main() {
    kernel<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
