#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Thread %d\n", threadIdx.x);
}

int main() {
    kernel<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
