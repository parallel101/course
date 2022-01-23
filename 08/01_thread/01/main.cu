#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Hello, world!\n");
}

int main() {
    kernel<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
