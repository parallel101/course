#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tnum = blockDim.x * gridDim.x;
    printf("Flattened Thread %d of %d\n", tid, tnum);
}

int main() {
    kernel<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
