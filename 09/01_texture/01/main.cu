#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "CudaAllocator.cuh"
#include "CudaArray.cuh"
#include "ticktock.h"
#include "writevdb.h"

__global__ void kernel(cudaTextureObject_t vel, cudaSurfaceObject_t loc, unsigned int n) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    float vel = tex3D<float>(vel, x, y, z);
    surf3Dwrite<float>(value, clr_nxt, x, y, z, cudaBoundaryModeTrap);
}

int main() {
    unsigned int n = 32;

    auto loc = CudaArray<float>::make({{n, n, n}, 0});
    auto loc_s = CudaSurface<float>::make(loc);
    auto vel = CudaArray<float>::make({{n, n, n}, 0});
    auto vel_t = CudaTexture<float>::make(vel);

    kernel<<<dim3(n / 8, n / 8, n / 8), dim3(8, 8, 8)>>>(vel_t.get(), loc_s.get(), n);

    std::vector<float> arr(n * n * n);
    clr.getArray().copyOut(arr.data());
    writevdb<float>("/tmp/a.vdb", n, n, n, arr.data());

    return 0;
}
