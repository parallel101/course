#pragma once

#include <array>
#include <memory>
#include <cuda_runtime.h>
#include "helper_cuda.h"


struct ctor_t {
};

static constexpr ctor_t ctor;

struct nocopy_t {
    nocopy_t() = default;
    nocopy_t(nocopy_t const &) = delete;
    nocopy_t &operator=(nocopy_t const &) = delete;
    nocopy_t(nocopy_t &&) = delete;
    nocopy_t &operator=(nocopy_t &&) = delete;
};


template <class T>
struct CudaArray {
    struct BuildArgs {
        std::array<unsigned int, 3> const dim{};
        cudaChannelFormatDesc desc{cudaCreateChannelDesc<T>()};  // or cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned)
        int flags{cudaArraySurfaceLoadStore}; // or 0
    };

protected:
    struct Impl {
        cudaArray *m_cuArray{};
        std::array<unsigned int, 3> m_dim{};

        explicit Impl(BuildArgs const &_args)
            : m_dim(_args.dim) {
            checkCudaErrors(cudaMalloc3DArray(&m_cuArray, &_args.desc, make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]), _args.flags));
        }

        void copyIn(T const *_data) {
            cudaMemcpy3DParms copy3DParams{};
            copy3DParams.srcPtr = make_cudaPitchedPtr((void *)_data, m_dim[0] * sizeof(T), m_dim[0], m_dim[1]);
            copy3DParams.dstArray = m_cuArray;
            copy3DParams.extent = make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]);
            copy3DParams.kind = cudaMemcpyHostToDevice;
            checkCudaErrors(cudaMemcpy3D(&copy3DParams));
        }

        void copyOut(T *_data) {
            cudaMemcpy3DParms copy3DParams{};
            copy3DParams.srcArray = m_cuArray;
            copy3DParams.dstPtr = make_cudaPitchedPtr((void *)_data, m_dim[0] * sizeof(T), m_dim[0], m_dim[1]);
            copy3DParams.extent = make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]);
            copy3DParams.kind = cudaMemcpyDeviceToHost;
            checkCudaErrors(cudaMemcpy3D(&copy3DParams));
        }

        ~Impl() {
            checkCudaErrors(cudaFreeArray(m_cuArray));
        }
    };

    std::shared_ptr<Impl> m_impl;

public:
    CudaArray(ctor_t, BuildArgs const &_args)
        : m_impl(std::make_shared<Impl>(_args)) {
    }

    void copyIn(T const *_data) const {
        m_impl->copyIn(_data);
    }

    void copyOut(T *_data) const {
        m_impl->copyOut(_data);
    }

    operator cudaArray *() const {
        return m_impl->m_cuArray;
    }
};

template <class T>
struct CudaSurface {
protected:
    struct Impl {
        cudaSurfaceObject_t m_cuSuf{};
        CudaArray<T> m_cuarr;

        explicit Impl(CudaArray<T> const &_cuarr)
            : m_cuarr(_cuarr) {
            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypeArray;

            resDesc.res.array.array = m_cuarr;
            checkCudaErrors(cudaCreateSurfaceObject(&m_cuSuf, &resDesc));
        }

        ~Impl() {
            checkCudaErrors(cudaDestroySurfaceObject(m_cuSuf));
        }
    };

    std::shared_ptr<Impl> m_impl;

public:
    CudaSurface(ctor_t, CudaArray<T> const &_cuarr)
        : m_impl(std::make_shared<Impl>(_cuarr)) {
    }

    CudaArray<T> getArray() const {
        return m_impl->m_cuarr;
    }

    cudaSurfaceObject_t get() const {
        return m_impl->m_cuSuf;
    }

    struct Accessor {
        cudaSurfaceObject_t m_cuSuf;

        template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>  // or cudaBoundaryModeZero, cudaBoundaryModeClamp
        __device__ __forceinline__ T read(int x, int y, int z) const {
            return surf3Dread<T>(m_cuSuf, x * sizeof(T), y, z, mode);
        }

        template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>  // or cudaBoundaryModeZero, cudaBoundaryModeClamp
        __device__ __forceinline__ void write(T val, int x, int y, int z) const {
            surf3Dwrite<T>(val, m_cuSuf, x * sizeof(T), y, z, mode);
        }
    };

    Accessor access() const {
        return {m_impl->m_cuSuf};
    }
};

template <class T>
struct CudaTexture {
    struct BuildArgs {
        cudaTextureAddressMode addressMode{cudaAddressModeClamp};  // or cudaAddressModeWrap
        cudaTextureFilterMode filterMode{cudaFilterModeLinear};   // or cudaFilterModePoint
        cudaTextureReadMode readMode{cudaReadModeElementType};  // or cudaReadModeNormalizedFloat
        bool normalizedCoords{false};
    };

protected:
    struct Impl {
        cudaTextureObject_t m_cuTex{};
        CudaArray<T> m_cuarr;

        explicit Impl(CudaArray<T> const &_cuarr, BuildArgs const &_args)
            : m_cuarr(_cuarr) {
            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = m_cuarr;

            cudaTextureDesc texDesc{};
            texDesc.addressMode[0] = _args.addressMode;
            texDesc.addressMode[1] = _args.addressMode;
            texDesc.addressMode[2] = _args.addressMode;
            texDesc.filterMode = _args.filterMode;
            texDesc.readMode = _args.readMode;
            texDesc.normalizedCoords = _args.normalizedCoords;

            checkCudaErrors(cudaCreateTextureObject(&m_cuTex, &resDesc, &texDesc, NULL));
        }

        ~Impl() {
            checkCudaErrors(cudaDestroyTextureObject(m_cuTex));
        }
    };

    std::shared_ptr<Impl> m_impl;

public:
    CudaTexture(ctor_t, CudaArray<T> const &_cuarr, BuildArgs const &_args = {})
        : m_impl(std::make_shared<Impl>(_cuarr, _args)) {
    }

    CudaArray<T> getArray() const {
        return m_impl->m_cuarr;
    }

    cudaTextureObject_t get() const {
        return m_impl->m_cuTex;
    }

    struct Accessor {
        cudaTextureObject_t m_cuTex;

        __device__ __forceinline__ T sample(float x, float y, float z) const {
            return tex3D<T>(m_cuTex, x, y, z);
        }
    };

    Accessor access() const {
        return {m_impl->m_cuTex};
    }
};

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
