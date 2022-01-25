#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/universal_vector.h>

template <class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 65536;
    float a = 3.14f;
    thrust::universal_vector<float> x(n);
    thrust::universal_vector<float> y(n);

    for (int i = 0; i < n; i++) {
        x[i] = std::rand() * (1.f / RAND_MAX);
        y[i] = std::rand() * (1.f / RAND_MAX);
    }

    parallel_for<<<n / 512, 128>>>(n, [a, x = x.data(), y = y.data()] __device__ (int i) {
        x[i] = a * x[i] + y[i];
    });
    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    return 0;
}
