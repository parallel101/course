#include "gaussblur.h"
#include <x86intrin.h>
#include <vector>
#include <cmath>

static void x_kernel_blur(Image &b, Image const &a, int nblur, float *__restrict kernel) {
    int nx = a.shape(0);
    int ny = a.shape(1);
    int ncomp = a.shape(2);
    b.reshape((size_t)nx, (size_t)ny, (size_t)ncomp);
    for (int comp = 0; comp < ncomp; comp++) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int xBase = 0; xBase < nx; xBase += 16) {
                _mm_prefetch(&a(xBase + 16, y, comp), _MM_HINT_T0);
                for (int x = xBase; x < xBase + 16; x += 4) {
                    __m128 res = _mm_setzero_ps();
                    for (int t = -nblur; t <= nblur; t++) {
                        __m128 tmp = _mm_loadu_ps(&a(x + t, y, comp));
                        res = _mm_fmadd_ps(tmp, _mm_set1_ps(kernel[nblur + t]), res);
                    }
                    _mm_stream_ps(&b(x, y, comp), res);
                }
            }
        }
    }
}

static void y_kernel_blur(Image &b, Image const &a, int nblur, float *__restrict kernel) {
    int nx = a.shape(0);
    int ny = a.shape(1);
    int ncomp = a.shape(2);
    b.reshape((size_t)nx, (size_t)ny, (size_t)ncomp);
    for (int comp = 0; comp < ncomp; comp++) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x += 32) {
            for (int y = 0; y < ny; y++) {
                _mm_prefetch(&a(x, y + nblur + 40, comp), _MM_HINT_T0);
                _mm_prefetch(&a(x + 16, y + nblur + 40, comp), _MM_HINT_T0);
                __m256 res[4];
#ifdef _MSC_VER
#pragma unroll 4
#else
#pragma GCC unroll 4
#endif
                for (int offset = 0; offset < 4; offset++) {
                    __m256 tmp = _mm256_load_ps(&a(x + offset * 8, y - nblur, comp));
                    res[offset] = _mm256_mul_ps(tmp, _mm256_set1_ps(kernel[0]));
                }
                for (int t = -nblur + 1; t <= nblur; t++) {
                    __m256 fac = _mm256_set1_ps(kernel[nblur + t]);
#ifdef _MSC_VER
#pragma unroll 4
#else
#pragma GCC unroll 4
#endif
                    for (int offset = 0; offset < 4; offset++) {
                        __m256 tmp = _mm256_load_ps(&a(x + offset * 8, y + t, comp));
                        res[offset] = _mm256_fmadd_ps(tmp, fac, res[offset]);
                    }
                }
#ifdef _MSC_VER
#pragma unroll 4
#else
#pragma GCC unroll 4
#endif
                for (int offset = 0; offset < 4; offset++) {
                    _mm256_stream_ps(&b(x + offset * 8, y, comp), res[offset]);
                }
            }
        }
    }
}

static std::vector<float> make_gaussian(int nblur, float sigma) {
    std::vector<float> ret(2 * nblur + 1);
    float factor = -0.5f / (sigma * sigma);
    for (int i = 0; i <= nblur; i++) {
        float tmp = std::exp((i * i) * factor);
        ret[nblur + i] = tmp;
        ret[nblur - i] = tmp;
    }
    float sum = 0.f;
    for (int i = 0; i <= 2 * nblur; i++) {
        sum += ret[i];
    }
    for (int i = 0; i <= 2 * nblur; i++) {
        ret[i] /= sum;
    }
    return ret;
}

void gaussblur(Image &a, int nblur, float sigma) {
    if (!nblur) return;
    Image b(a.shape());
    auto kernel = make_gaussian(nblur, sigma);
    x_kernel_blur(b, a, nblur, kernel.data());
    y_kernel_blur(a, b, nblur, kernel.data());
}
