#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>

template <class T>
struct CudaAllocator {
    using value_type = T;

    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }

    template <class ...Args>
    void construct(T *p, Args &&...args) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }
};

template <int N, class T>
__global__ void kernel(T *arr) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < N; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}

int main() {
    constexpr int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);

    kernel<n><<<32, 128>>>(arr.data());

    checkCudaErrors(cudaDeviceSynchronize());
    for (int i = 0; i < n; i++) {
        printf("arr[%d]: %d\n", i, arr[i]);
    }

    return 0;
}
