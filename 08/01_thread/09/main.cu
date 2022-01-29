#include <cstdio>
#include <cuda_runtime.h>

__device__ void say_hello();  // 声明

__global__ void kernel() {
    say_hello();
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
