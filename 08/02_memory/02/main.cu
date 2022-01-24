#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel(int *pret) {
    *pret = 42;
}

int main() {
    int ret = 0;
    kernel<<<1, 1>>>(&ret);
    cudaDeviceSynchronize();
    printf("%d\n", ret);
    return 0;
}
