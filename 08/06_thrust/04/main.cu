#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/for_each.h>

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

    thrust::for_each(x_host.begin(), x_host.end(), [] (float &x) {
        x = std::rand() * (1.f / RAND_MAX);
    });
    thrust::for_each(y_host.begin(), y_host.end(), [] (float &y) {
        y = std::rand() * (1.f / RAND_MAX);
    });

    thrust::device_vector<float> x_dev = x_host;
    thrust::device_vector<float> y_dev = x_host;

    thrust::for_each(x_dev.begin(), x_dev.end(), [] __device__ (float &x) {
        x += 100.f;
    });

    thrust::for_each(x_dev.cbegin(), x_dev.cend(), [] __device__ (float const &x) {
        printf("%f\n", x);
    });

    parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x_dev.data(), y_dev = y_dev.data()] __device__ (int i) {
        x_dev[i] = a * x_dev[i] + y_dev[i];
    });

    x_host = x_dev;

    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x_host[i]);
    }

    return 0;
}
