#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel(int *pret) {
    *pret = 42;
}

int main() {
    int ret = 0;
    kernel<<<1, 1>>>(&ret);
    cudaError_t err = cudaDeviceSynchronize();
    printf("error code: %d\n", err);
    printf("error name: %s\n", cudaGetErrorName(err));
    printf("%d\n", ret);
    return 0;
}
