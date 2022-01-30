#pragma once

#include <cstddef>
#include <utility>
#include <cuda_runtime.h>
#include "helper_cuda.h"

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

    constexpr bool operator==(CudaAllocator<T> const &other) const {
        return this == &other;
    }
};
