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
    if (x >= n || y >= n || z >= n) return;
    float4 vel = tex3D<float4>(texVel, x, y, z);
    float4 loc = make_float4(x + 0.5f, y + 0.5f, z + 0.5f, 42.f) - vel;
    surf3Dwrite<float4>(loc, sufLoc, x * sizeof(float4), y, z, cudaBoundaryModeTrap);
}

int main() {
    unsigned int n = 2;

    auto arrLoc = CudaArray<float4>::make({{n, n, n}});//, cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat)});
    auto sufLoc = CudaSurface<float4>::make(arrLoc);
    auto arrVel = CudaArray<float4>::make({{n, n, n}});//, cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat)});
    auto sufVel = CudaSurface<float4>::make(arrVel);
    auto texVel = CudaTexture<float4>::make(arrVel);

    std::vector<float4> cpuVel(n * n * n);
    for (unsigned int z = 0; z < n; z++) {
        for (unsigned int y = 0; y < n; y++) {
            for (unsigned int x = 0; x < n; x++) {
                cpuVel[x + n * (y + n * z)] = make_float4(0.1f, 0.f, 0.f, 0.f);
            }
        }
    }
    arrVel.copyIn(cpuVel.data());

    kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(texVel.get(), sufLoc.get(), n);

    std::vector<float4> cpuLoc(n * n * n);
    arrLoc.copyOut(cpuLoc.data());
    for (unsigned int z = 0; z < n; z++) {
        for (unsigned int y = 0; y < n; y++) {
            for (unsigned int x = 0; x < n; x++) {
                float4 val = cpuLoc[x + n * (y + n * z)];
                printf("%u,%u,%u: %f,%f,%f,%f\n", x, y, z, val.x, val.y, val.z, val.w);
            }
        }
    }
    //writevdb<float, 3>("/tmp/a.vdb", n, n, n, arr.data());

    return 0;
}
