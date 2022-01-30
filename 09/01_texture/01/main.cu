#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "CudaArray.cuh"
#include "ticktock.h"
#include "writevdb.h"

__global__ void advect_kernel(CudaTexture<float4>::Accessor texVel, CudaSurface<float4>::Accessor sufLoc, unsigned int n) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    float4 vel = texVel.sample(x, y, z);
    float4 loc = make_float4(x + 0.5f, y + 0.5f, z + 0.5f, 42.f) - vel;
    sufLoc.write(loc, x, y, z);
}

__global__ void resample_kernel(CudaSurface<float4>::Accessor sufLoc, CudaTexture<float4>::Accessor texClr, CudaSurface<float4>::Accessor sufClrNext, unsigned int n) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;

    float4 loc = sufLoc.read(x, y, z);
    float4 clr = texClr.sample(loc.x, loc.y, loc.z);
    sufClrNext.write(clr, x, y, z);
}

template <class T>
struct CudaAST {
    CudaArray<T> arr;
    CudaSurface<T> suf;
    CudaTexture<T> tex;

    CudaAST(ctor_t, typename CudaArray<T>::BuildArgs const &_arrArgs, typename CudaTexture<T>::BuildArgs const &_texArgs = {})
        : arr(ctor, _arrArgs)
        , suf(ctor, arr)
        , tex(ctor, arr, _texArgs)
    {
    }
};

struct SmokeSim {
    nocopy_t nocopy;

    unsigned int n;
    CudaAST<float4> loc;
    CudaAST<float4> vel;
    CudaAST<float4> velNext;
    CudaAST<float4> clr;
    CudaAST<float4> clrNext;

    SmokeSim(unsigned int _n)
    : n(_n)
    , loc(ctor, {{n, n, n}})
    , vel(ctor, {{n, n, n}})
    , velNext(ctor, {{n, n, n}})
    , clr(ctor, {{n, n, n}})
    , clrNext(ctor, {{n, n, n}})
    {}

    void advection() {
        advect_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(vel.tex.access(), loc.suf.access(), n);
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(loc.suf.access(), clr.tex.access(), clrNext.suf.access(), n);
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(loc.suf.access(), vel.tex.access(), velNext.suf.access(), n);

        std::swap(vel, velNext);
        std::swap(clr, clrNext);
    }
};

int main() {
    unsigned int n = 32;

    SmokeSim sim(n);

    std::vector<float4> cpuVel(n * n * n);
    for (unsigned int z = 0; z < n; z++) {
        for (unsigned int y = 0; y < n; y++) {
            for (unsigned int x = 0; x < n; x++) {
                cpuVel[x + n * (y + n * z)] = make_float4(0.1f, 0.1f, 0.f, 0.f);
            }
        }
    }
    sim.vel.arr.copyIn(cpuVel.data());

    std::vector<float4> cpuLoc(n * n * n);
    for (int i = 0; i < 1024; i++) {
        sim.advection();

        sim.loc.arr.copyOut(cpuLoc.data());
        //for (unsigned int z = 0; z < n; z++) {
            //for (unsigned int y = 0; y < n; y++) {
                //for (unsigned int x = 0; x < n; x++) {
                    //float4 val = cpuLoc[x + n * (y + n * z)];
                    //printf("%u,%u,%u: %f,%f,%f,%f\n", x, y, z, val.x, val.y, val.z, val.w);
                //}
            //}
        //}
        writevdb<float, 1>("/tmp/a" + std::to_string(1000 + i).substr(1) + ".vdb", cpuLoc.data(), n, n, n, sizeof(float4));
    }

    return 0;
}
