#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);

    for (int i = 0; i < n; i++) {
        x_host[i] = std::rand() * (1.f / RAND_MAX);
        y_host[i] = std::rand() * (1.f / RAND_MAX);
    }

    thrust::device_vector<float> x_dev = x_host;
    thrust::device_vector<float> y_dev = x_host;

    parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x_dev.data(), y_dev = y_dev.data()] __device__ (int i) {
        x_dev[i] = a * x_dev[i] + y_dev[i];
    });

    x_host = x_dev;

    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x_host[i]);
    }

    return 0;
}
