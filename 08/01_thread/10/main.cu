#include <cstdio>
#include <cuda_runtime.h>

__global__ void another() {
    printf("another: Thread %d of %d\n", threadIdx.x, blockDim.x);
}

__global__ void kernel() {
    printf("kernel: Thread %d of %d\n", threadIdx.x, blockDim.x);
    int numthreads = threadIdx.x * threadIdx.x + 1;
    another<<<1, numthreads>>>();
    printf("kernel: called another with %d threads\n", numthreads);
}

int main() {
    kernel<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
