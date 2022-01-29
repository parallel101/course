#include <cstdio>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "ticktock.h"

struct DisableCopy {
    DisableCopy() = default;
    DisableCopy(DisableCopy const &) = delete;
    DisableCopy &operator=(DisableCopy const &) = delete;
    DisableCopy(DisableCopy &&) = delete;
    DisableCopy &operator=(DisableCopy &&) = delete;
};

template <class T>
struct CudaArray : DisableCopy {
    cudaArray *m_cuArray{};
    std::array<unsigned int, 3> m_dim{};

    struct BuildArgs {
        std::array<unsigned int, 3> const _dim;
        int flags = 0; // or cudaArraySurfaceLoadStore
    };

    explicit CudaArray(BuildArgs _args) : m_dim(_args.dim) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();  // or cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned)
        checkCudaErrors(cudaMalloc3DArray(&m_cuArray, &channelDesc, make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]), _args.flags));
    }

    CudaArray &assign(T *_data) {
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcPtr = make_cudaPitchedPtr((void *)_data, m_dim[0] * sizeof(T), m_dim[1], m_dim[2]);
        copy3DParams.dstArray = m_cuArray;
        copy3DParams.extent = make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]);
        copy3DParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
        return *this;
    }

    ~CudaArray() {
        checkCudaErrors(cudaFreeArray(m_cuArray));
    }

    operator cudaArray *() const {
        return m_cuArray;
    }
};

template <class T>
struct CudaSurface : DisableCopy {
    cudaSurfaceObject_t m_cuSuf{};
    CudaArray<T> m_cuarr;

    explicit CudaSurface(typename CudaArray<T>::BuildArgs _cuarrArgs) : m_cuarr(_cuarrArgs) {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;

        resDesc.res.array.array = m_cuarr.m_cuArray;
        cudaCreateSurfaceObject(&m_cuSuf, &resDesc);
    }

    ~CudaTexture() {
        checkCudaErrors(cudaDestroySurfaceObject(m_cuSuf));
    }

    CudaArray<T> &array() {
        return m_cuarr;
    }

    operator cudaSurfaceObject_t() const {
        return m_cuSuf;
    }
};

template <class T>
struct CudaTexture : DisableCopy {
    cudaTextureObject_t m_cuTex{};
    CudaArray<T> m_cuarr;

    explicit CudaTexture(typename CudaArray<T>::BuildArgs _cuarrArgs) : m_cuarr(_cuarrArgs) {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_cuarr.m_cuArray;

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp; // or cudaAddressModeWrap
        texDesc.addressMode[1] = cudaAddressModeClamp; // or cudaAddressModeWrap
        texDesc.addressMode[2] = cudaAddressModeClamp; // or cudaAddressModeWrap
        texDesc.filterMode = cudaFilterModePoint;      // or cudaFilterModeLinear
        texDesc.readMode = cudaReadModeElementType;    // or cudaReadModeNormalizedFloat
        texDesc.normalizedCoords = false;              // or true

        checkCudaErrors(cudaCreateTextureObject(&m_cuTex, &resDesc, &texDesc, NULL));
    }

    ~CudaTexture() {
        checkCudaErrors(cudaDestroyTextureObject(m_cuTex));
    }

    CudaArray<T> &array() {
        return m_cuarr;
    }

    operator cudaTextureObject_t() const {
        return m_cuTex;
    }
};

__global__ void kernel(cudaSurfaceObject_t out, cudaTextureObject_t in) {
    int x = 0, y = 0;
    float fx = 0, fy = 0, fz = 0;
    float value = tex3D<float>(in, fx, fy, fz);
    value += 1;
    surf2Dwrite(value, out, x, y);
    // or cudaBoundaryModeTrap, cudaBoundaryModeClamp
}

int main() {
    CudaSurface<float> out({{1, 1, 1}, cudaArraySurfaceLoadStore});
    CudaTexture<float> in({{1, 1, 1}, 0});
    return 0;
}
