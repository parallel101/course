#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "helper_cuda.h"


struct CudaEvent;

struct CudaStream {
protected:
    struct Impl {
        cudaStream_t m_stream;

        Impl() {
            checkCudaErrors(cudaStreamCreate(&m_stream));
        }

        ~Impl() {
            checkCudaErrors(cudaStreamDestroy(m_stream));
        }
    };

    std::shared_ptr<Impl> m_impl;

public:
    CudaStream()
        : m_impl(std::make_shared<Impl>()) {
    }

    cudaStream_t get() const {
        return m_impl ? m_impl->m_stream : 0;
    }

    void synchronize() const {
        checkCudaErrors(cudaStreamSynchronize(get()));
    }

    inline CudaEvent event() const;

    inline void wait(CudaEvent const &event) const;

    operator cudaStream_t() const {
        return get();
    }
};


struct CudaEvent {
protected:
    struct Impl {
        cudaEvent_t m_event;

        Impl() {
            checkCudaErrors(cudaEventCreate(&m_event));
        }

        ~Impl() {
            checkCudaErrors(cudaEventDestroy(m_event));
        }
    };

    std::shared_ptr<Impl> m_impl;

public:
    CudaEvent()
        : m_impl(std::make_shared<Impl>()) {
    }

    cudaEvent_t get() const {
        return m_impl->m_event;
    }

    operator cudaEvent_t() const {
        return get();
    }

    void synchronize() const {
        checkCudaErrors(cudaEventSynchronize(get()));
    }

    void record() const {
        checkCudaErrors(cudaEventRecord(get()));
    }

    void record(CudaStream const &stream) const {
        checkCudaErrors(cudaEventRecord(get(), stream.get()));
    }

    float elapsedTime(CudaEvent const &other) const {
        float res;
        checkCudaErrors(cudaEventElapsedTime(&res, get(), other.get()));
        return res;
    }
};

CudaEvent CudaStream::event() const {
    CudaEvent e;
    e.record(*this);
    return e;
}

void CudaStream::wait(CudaEvent const &event) const {
    checkCudaErrors(cudaStreamWaitEvent(get(), event.get()));
}
