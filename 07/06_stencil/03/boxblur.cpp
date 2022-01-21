#include "boxblur.h"
#include <x86intrin.h>

void xblur(Image &b, Image const &a, int nblur) {
    if (!nblur) { b = a; return; }
    int nx = a.shape(0);
    int ny = a.shape(1);
    int ncomp = a.shape(2);
    b.reshape((size_t)nx, (size_t)ny, (size_t)ncomp);
    __m128 factor = _mm_set1_ps(1.f / (2 * nblur + 1));
    for (int comp = 0; comp < ncomp; comp++) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int xBase = 0; xBase < nx; xBase += 16) {
                _mm_prefetch(&a(xBase + 16, y, comp), _MM_HINT_T0);
                for (int x = xBase; x < xBase + 16; x += 4) {
                    __m128 res = _mm_setzero_ps();
                    for (int t = -nblur; t <= nblur; t++) {
                        res = _mm_add_ps(res, _mm_loadu_ps(&a(x + t, y, comp)));
                    }
                    _mm_stream_ps(&b(x, y, comp), _mm_mul_ps(res, factor));
                }
            }
        }
    }
}

void yblur(Image &b, Image const &a, int nblur) {
    if (!nblur) { b = a; return; }
    int nx = a.shape(0);
    int ny = a.shape(1);
    int ncomp = a.shape(2);
    b.reshape((size_t)nx, (size_t)ny, (size_t)ncomp);
    __m256 factor = _mm256_set1_ps(1.f / (2 * nblur + 1));
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
                    res[offset] = _mm256_load_ps(&a(x + offset * 8, y - nblur, comp));
                }
                for (int t = -nblur + 1; t <= nblur; t++) {
#ifdef _MSC_VER
#pragma unroll 4
#else
#pragma GCC unroll 4
#endif
                    for (int offset = 0; offset < 4; offset++) {
                        res[offset] = _mm256_add_ps(res[offset],
                            _mm256_load_ps(&a(x + offset * 8, y + t, comp)));
                    }
                }
#ifdef _MSC_VER
#pragma unroll 4
#else
#pragma GCC unroll 4
#endif
                for (int offset = 0; offset < 4; offset++) {
                    res[offset] = _mm256_mul_ps(res[offset], factor);
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

void boxblur(Image &a, int nxblur, int nyblur) {
    if (!nxblur && !nyblur) return;
    Image b(a.shape());
    xblur(b, a, nxblur);
    yblur(a, b, nyblur);
}
