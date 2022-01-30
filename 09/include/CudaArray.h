#pragma once

#include <array>
#include <memory>
#include <cuda_runtime.h>
#include "helper_cuda.h"


template <class T>
class CudaArray {
    struct BuildArgs {
        std::array<unsigned int, 3> const dim{};
        int flags{0}; // or cudaArraySurfaceLoadStore
    };

    struct Impl {
        cudaArray *m_cuArray{};
        std::array<unsigned int, 3> m_dim{};

        explicit Impl(BuildArgs const &_args)
            : m_dim(_args.dim) {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();  // or cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned)
            checkCudaErrors(cudaMalloc3DArray(&m_cuArray, &channelDesc, make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]), _args.flags));
        }

        void copyIn(T const *_data) {
            cudaMemcpy3DParms copy3DParams{};
            copy3DParams.srcPtr = make_cudaPitchedPtr((void *)_data, m_dim[0] * sizeof(T), m_dim[1], m_dim[2]);
            copy3DParams.dstArray = m_cuArray;
            copy3DParams.extent = make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]);
            copy3DParams.kind = cudaMemcpyHostToDevice;
            checkCudaErrors(cudaMemcpy3D(&copy3DParams));
        }

        void copyOut(T *_data) {
            cudaMemcpy3DParms copy3DParams{};
            copy3DParams.srcArray = m_cuArray;
            copy3DParams.dstPtr = make_cudaPitchedPtr((void *)_data, m_dim[0] * sizeof(T), m_dim[1], m_dim[2]);
            copy3DParams.extent = make_cudaExtent(m_dim[0], m_dim[1], m_dim[2]);
            copy3DParams.kind = cudaMemcpyDeviceToHost;
            checkCudaErrors(cudaMemcpy3D(&copy3DParams));
        }

        ~Impl() {
            checkCudaErrors(cudaFreeArray(m_cuArray));
        }
    };

    std::shared_ptr<Impl> impl;

public:
    static CudaArray make(BuildArgs const &_args) {
        CudaArray that;
        that.impl = std::make_shared<Impl>(_args);
        return that;
    }

    void copyIn(T const *_data) const {
        impl->copyIn(_data);
    }

    void copyOut(T *_data) const {
        impl->copyOut(_data);
    }

    operator cudaArray *() const {
        return impl->m_cuArray;
    }
};

template <class T>
class CudaSurface {
    struct Impl {
        cudaSurfaceObject_t m_cuSuf{};
        CudaArray<T> m_cuarr;

        explicit Impl(CudaArray<T> const &_cuarr)
            : m_cuarr(_cuarr) {
            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypeArray;

            resDesc.res.array.array = m_cuarr;
            cudaCreateSurfaceObject(&m_cuSuf, &resDesc);
        }

        ~Impl() {
            checkCudaErrors(cudaDestroySurfaceObject(m_cuSuf));
        }
    };

    std::shared_ptr<Impl> impl;

public:
    static CudaSurface make(CudaArray<T> const &_cuarr) {
        CudaSurface that;
        that.impl = std::make_shared<Impl>(_cuarr);
        return that;
    }

    CudaArray<T> getArray() const {
        return impl->m_cuarr;
    }

    operator cudaSurfaceObject_t() const {
        return impl->m_cuSuf;
    }
};

template <class T>
class CudaTexture {
    struct BuildArgs {
        cudaTextureAddressMode addressMode{cudaAddressModeClamp};  // or cudaAddressModeWrap
        cudaTextureFilterMode filterMode{cudaFilterModeLinear};   // or cudaFilterModePoint
        cudaTextureReadMode readMode{cudaReadModeElementType};  // or cudaReadModeNormalizedFloat
        bool normalizedCoords{false};
    };

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

    std::shared_ptr<Impl> impl;

public:
    static CudaTexture make(CudaArray<T> const &_cuarr, BuildArgs const &_args = {}) {
        CudaTexture that;
        that.impl = std::make_shared<Impl>(_cuarr, _args);
        return that;
    }

    CudaArray<T> getArray() const {
        return impl->m_cuarr;
    }

    operator cudaTextureObject_t() const {
        return impl->m_cuTex;
    }
};
