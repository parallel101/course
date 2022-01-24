#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *pret) {
    *pret = 42;
}

int main() {
    int *pret;
    checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
    kernel<<<1, 1>>>(pret);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(pret);
    return 0;
}
