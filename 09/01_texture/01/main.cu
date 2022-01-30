#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "CudaArray.h"
#include "ticktock.h"
#include "ycmcudahelp.h"

__global__ void kernel(cudaSurfaceObject_t out, cudaTextureObject_t in) {
    int x = 0, y = 0, z = 0;
    float fx = 0, fy = 0, fz = 0;
    float value = tex3D<float>(in, fx, fy, fz);
    value = 3;
    surf3Dwrite<float>(value, out, x, y, z, cudaBoundaryModeTrap);  // or cudaBoundaryModeZero, cudaBoundaryModeClamp
}

int main() {
    unsigned int n = 2;
    auto out = CudaSurface<float>::make(CudaArray<float>::make({{n, n, n}, cudaArraySurfaceLoadStore}));
    auto in = CudaTexture<float>::make(CudaArray<float>::make({{n, n, n}, 0}));
    kernel<<<1, 1>>>(out, in);
    std::vector<float> arr(n * n * n);
    out.getArray().copyOut(arr.data());
    for (int i = 0; i < arr.size(); i++) {
        printf("%f\n", arr[i]);
    }
    return 0;
}
