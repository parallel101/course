#include <cstdio>
#include <cuda_runtime.h>

__global__ int kernel() {
    return 42;
}

int main() {
    int ret = kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("%d\n", ret);
    return 0;
}
