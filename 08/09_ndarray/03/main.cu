#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

template <int blockSize, class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny) {
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;
    if (x >= nx || y >= ny) return;
    __shared__ T tmp[blockSize * blockSize];
    int rx = blockIdx.y * blockSize + threadIdx.x;
    int ry = blockIdx.x * blockSize + threadIdx.y;
    tmp[threadIdx.y * blockSize + threadIdx.x] = in[ry * nx + rx];
    __syncthreads();
    out[y * nx + x] = tmp[threadIdx.x * blockSize + threadIdx.y];
}

int main() {
    int nx = 1<<14, ny = 1<<14;
    std::vector<int, CudaAllocator<int>> in(nx * ny);
    std::vector<int, CudaAllocator<int>> out(nx * ny);

    for (int i = 0; i < nx * ny; i++) {
        in[i] = i;
    }

    TICK(parallel_transpose);
    parallel_transpose<32><<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>
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
