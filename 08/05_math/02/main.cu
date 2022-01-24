#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

template <class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 1<<25;
    std::vector<float, CudaAllocator<float>> gpu(n);
    std::vector<float> cpu(n);

    TICK(cpu_sinf);
    for (int i = 0; i < n; i++) {
        cpu[i] = sinf(i);
    }
    TOCK(cpu_sinf);

    TICK(gpu_sinf);
    parallel_for<<<n / 512, 128>>>(n, [gpu = gpu.data()] __device__ (int i) {
        gpu[i] = sinf(i);
    });
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(gpu_sinf);

    //for (int i = 0; i < n; i++) {
        //printf("diff %d = %f\n", i, gpu[i] - cpu[i]);
    //}

    return 0;
}
