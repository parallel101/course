#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

template <class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    out[y * nx + x] = in[x * nx + y];
}

int main() {
    int nx = 1<<14, ny = 1<<14;
    std::vector<int, CudaAllocator<int>> in(nx * ny);
    std::vector<int, CudaAllocator<int>> out(nx * ny);

    for (int i = 0; i < nx * ny; i++) {
        in[i] = i;
    }

    TICK(parallel_transpose);
    parallel_transpose<<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>
        (out.data(), in.data(), nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_transpose);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            if (out[y * nx + x] != in[x * nx + y]) {
                printf("Wrong At x=%d,y=%d: %d != %d\n", x, y,
                       out[y * nx + x], in[x * nx + y]);
                return -1;
            }
        }
    }

    printf("All Correct!\n");
    return 0;
}
