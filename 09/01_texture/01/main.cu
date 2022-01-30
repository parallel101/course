#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "CudaAllocator.cuh"
#include "CudaArray.cuh"
#include "ticktock.h"
#include "writevdb.h"

__global__ void kernel(cudaTextureObject_t texVel, cudaSurfaceObject_t sufLoc, unsigned int n) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    float4 vel = tex3D<float4>(texVel, x, y, z);
    float4 loc = make_float4(x + 0.5f, y + 0.5f, z + 0.5f, 0.f) - vel;
    surf3Dwrite<float4>(loc, sufLoc, x, y, z, cudaBoundaryModeTrap);
}

int main() {
    unsigned int n = 4;

    auto arrLoc = CudaArray<float4>::make({{n, n, n}, 0});
    auto sufLoc = CudaSurface<float4>::make(arrLoc);
    auto arrVel = CudaArray<float4>::make({{n, n, n}, 0});
    auto texVel = CudaTexture<float4>::make(arrVel);

    std::vector<float4> cpuVel(n * n * n);
    for (int z = 0; z < n; z++) {
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                cpuVel[x + n * (y + n * z)] = make_float4(1.f, 0.f, 0.f, 0.f);
            }
        }
    }
    arrVel.copyIn(cpuVel.data());

    kernel<<<dim3(n / 8, n / 8, n / 8), dim3(8, 8, 8)>>>(texVel.get(), sufLoc.get(), n);

    std::vector<float4> cpuLoc(n * n * n);
    arrLoc.copyOut(cpuLoc.data());
    for (int z = 0; z < n; z++) {
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                printf("%d,%d,%d: %f\n", x, y, z, cpuLoc[x + n * (y + n * z)]);
            }
        }
    }
    //writevdb<float, 3>("/tmp/a.vdb", n, n, n, arr.data());

    return 0;
}
