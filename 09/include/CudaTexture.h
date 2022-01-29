#pragma once


#include "helper_cuda.h"
#include <cuda_runtime.h>


struct CudaTexture {
    cudaTextureObject_t tex;

    CudaTexture(CudaTexture const &) = delete;
    CudaTexture(CudaTexture &&) = default;
    CudaTexture &operator=(CudaTexture const &) = delete;
    CudaTexture &operator=(CudaTexture &&) = default;

    template <class T>
    CudaTexture(T *dataDev, int width, int height) {
        cudaTextureObject_t tex;
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = dataDev;
        resDesc.res.pitch2D.width = width;
        resDesc.res.pitch2D.height = height;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
        resDesc.res.pitch2D.pitchInBytes = width * sizeof(T);
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    }

    ~CudaTexture() {
        checkCudaErrors(cudaDestroyTextureObject(tex));
    }

    constexpr operator cudaTextureObject_t() const {
        return tex;
    }
};
