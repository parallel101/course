#include <cuda_runtime.h>
#include <nvfunctional>
#include "cudapp.cuh"

using namespace cudapp;

extern "C" __global__ void kernel(int x) {
    printf("内核参数 x = %d\n", x);
    printf("线程编号 (%d, %d)\n", blockIdx.x, threadIdx.x);
}

int main() {
    int x = 42;
    kernel<<<3, 4, 0, 0>>>(x);

    void *args[] = {&x};
    CHECK_CUDA(cudaLaunchKernel((const void *)kernel, dim3(3), dim3(4), args, 0, 0));

    cudaLaunchConfig_t cfg{};
    cfg.blockDim = dim3(3);
    cfg.gridDim = dim3(4);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = 0;
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    CHECK_CUDA(cudaLaunchKernelEx(&cfg, kernel, x));

    const char *name;
    CHECK_CUDA(cudaFuncGetName(&name, (const void *)kernel));

    CudaStream::nullStream().join();
    return 0;
}
