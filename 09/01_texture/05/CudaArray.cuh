#pragma once

#include <cuda_runtime.h>
#include "helper_cuda.h"

struct DisableCopy {
    DisableCopy() = default;
    DisableCopy(DisableCopy const &) = delete;
    DisableCopy &operator=(DisableCopy const &) = delete;
};

template <class T>
struct CudaArray : DisableCopy {
    cudaArray *m_cuArray{};
    uint3 m_dim{};

    explicit CudaArray(uint3 const &_dim)
        : m_dim(_dim) {
        cudaExtent extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        checkCudaErrors(cudaMalloc3DArray(&m_cuArray, &channelDesc, extent, cudaArraySurfaceLoadStore));
    }

    void copyIn(T const *_data) {
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcPtr = make_cudaPitchedPtr((void *)_data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.dstArray = m_cuArray;
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }

    void copyOut(T *_data) {
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcArray = m_cuArray;
        copy3DParams.dstPtr = make_cudaPitchedPtr((void *)_data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3DParams.kind = cudaMemcpyDeviceToHost;
        checkCudaErrors(cudaMemcpy3D(&copy3DParams));
    }

    cudaArray *getArray() const {
        return m_cuArray;
    }

    ~CudaArray() {
        checkCudaErrors(cudaFreeArray(m_cuArray));
    }
};

template <class T>
struct CudaSurfaceAccessor {
    cudaSurfaceObject_t m_cuSuf;

    template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ T read(int x, int y, int z) const {
        return surf3Dread<T>(m_cuSuf, x * sizeof(T), y, z, mode);
    }

    template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ void write(T val, int x, int y, int z) const {
        surf3Dwrite<T>(val, m_cuSuf, x * sizeof(T), y, z, mode);
    }
};

template <class T>
struct CudaSurface : CudaArray<T> {
    cudaSurfaceObject_t m_cuSuf{};

    explicit CudaSurface(uint3 const &_dim)
        : CudaArray<T>(_dim) {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;

        resDesc.res.array.array = CudaArray<T>::getArray();
        checkCudaErrors(cudaCreateSurfaceObject(&m_cuSuf, &resDesc));
    }

    cudaSurfaceObject_t getSurface() const {
        return m_cuSuf;
    }

    CudaSurfaceAccessor<T> accessSurface() const {
        return {m_cuSuf};
    }

    ~CudaSurface() {
        checkCudaErrors(cudaDestroySurfaceObject(m_cuSuf));
    }
};

template <class T>
struct CudaTextureAccessor {
    cudaTextureObject_t m_cuTex;

    __device__ __forceinline__ T sample(float x, float y, float z) const {
        return tex3D<T>(m_cuTex, x, y, z);
    }
};

template <class T>
struct CudaTexture : CudaSurface<T> {
    struct Parameters {
        cudaTextureAddressMode addressMode{cudaAddressModeClamp};
        cudaTextureFilterMode filterMode{cudaFilterModeLinear};
        cudaTextureReadMode readMode{cudaReadModeElementType};
        bool normalizedCoords{false};
    };

    cudaTextureObject_t m_cuTex{};

    explicit CudaTexture(uint3 const &_dim, Parameters const &_args = {})
        : CudaSurface<T>(_dim) {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = CudaSurface<T>::getArray();

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = _args.addressMode;
        texDesc.addressMode[1] = _args.addressMode;
        texDesc.addressMode[2] = _args.addressMode;
        texDesc.filterMode = _args.filterMode;
        texDesc.readMode = _args.readMode;
        texDesc.normalizedCoords = _args.normalizedCoords;

        checkCudaErrors(cudaCreateTextureObject(&m_cuTex, &resDesc, &texDesc, NULL));
    }

    cudaTextureObject_t getTexture() const {
        return m_cuTex;
    }

    CudaTextureAccessor<T> accessTexture() const {
        return {m_cuTex};
    }

    ~CudaTexture() {
        checkCudaErrors(cudaDestroyTextureObject(m_cuTex));
    }
};
