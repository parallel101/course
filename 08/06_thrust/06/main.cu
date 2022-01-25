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

    auto float_rand = [] {
        return std::rand() * (1.f / RAND_MAX);
    };
    thrust::generate(x_host.begin(), x_host.end(), float_rand);
    thrust::generate(y_host.begin(), y_host.end(), float_rand);

    thrust::device_vector<float> x_dev = x_host;
    thrust::device_vector<float> y_dev = x_host;

    thrust::for_each(
        thrust::make_zip_iterator(x_dev.begin(), y_dev.cbegin()),
        thrust::make_zip_iterator(x_dev.end(), y_dev.cend()),
        [a] __device__ (auto const &tup) {
        auto &x = thrust::get<0>(tup);
        auto const &y = thrust::get<1>(tup);
        x = a * x + y;
    });

    x_host = x_dev;

    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x_host[i]);
    }

    return 0;
}
