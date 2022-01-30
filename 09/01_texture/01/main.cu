#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "CudaArray.h"
#include "ticktock.h"
#include "writevdb.h"

__global__ void kernel(cudaSurfaceObject_t vel, cudaTextureObject_t clr, unsigned int n) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    float value = tex3D<float>(in, x, y, z);
    value = 3;
    surf3Dwrite<float>(value, out, x, y, z, cudaBoundaryModeTrap);  // or cudaBoundaryModeZero, cudaBoundaryModeClamp
}

int main() {
    unsigned int n = 32;

    auto clr = CudaArray<float>::make({{n, n, n}, 0});
    auto clr_t = CudaTexture<float>::make(clr);
    auto vel = CudaArray<float>::make({{n, n, n}, 0});
    auto vel_s = CudaSurface<float>::make(vel);

    kernel<<<dim3(n / 8, n / 8, n / 8), dim3(8, 8, 8)>>>(vel_s, clr_t, n);

    std::vector<float> arr(n * n * n);
    out.getArray().copyOut(arr.data());
    writevdb<float>("/tmp/a.vdb", n, n, n, arr.data());

    return 0;
}
