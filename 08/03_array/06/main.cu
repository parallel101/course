#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *arr, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > n) return;
    arr[i] = i;
}

int main() {
    int n = 65535;
    int *arr;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));

    int nthreads = 128;
    int nblocks = (n + nthreads + 1) / nthreads;
    kernel<<<nblocks, nthreads>>>(arr, n);

    checkCudaErrors(cudaDeviceSynchronize());
    for (int i = 0; i < n; i++) {
        printf("arr[%d]: %d\n", i, arr[i]);
    }

    cudaFree(arr);
    return 0;
}
