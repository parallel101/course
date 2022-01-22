#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>
#include "ndarray.h"

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr size_t nx = 1<<13;
constexpr size_t ny = 1<<13;
constexpr int nblur = 8;

ndarray<2, float, 16> a(nx, ny);
ndarray<2, float> b(nx, ny);

void BM_copy(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                b(x, y) = a(x, y);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_copy);

void BM_copy_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x += 4) {
                _mm_stream_ps(&b(x, y), _mm_load_ps(&a(x, y)));
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_copy_streamed);

void BM_x_blur(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float res = 0;
                for (int t = -nblur; t <= nblur; t++) {
                    res += a(x + t, y);
                }
                b(x, y) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur);

void BM_x_blur_prefetched(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                _mm_prefetch(&a(x + 16, y), _MM_HINT_T0);
                float res = 0;
                for (int t = -nblur; t <= nblur; t++) {
                    res += a(x + t, y);
                }
                b(x, y) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur_prefetched);

void BM_x_blur_tiled_prefetched(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int xBase = 0; xBase < nx; xBase += 16) {
                _mm_prefetch(&a(xBase + 16, y), _MM_HINT_T0);
                for (int x = xBase; x < xBase + 16; x++) {
                    float res = 0;
                    for (int t = -nblur; t <= nblur; t++) {
                        res += a(x + t, y);
                    }
                    b(x, y) = res;
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur_tiled_prefetched);

void BM_x_blur_tiled_prefetched_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int xBase = 0; xBase < nx; xBase += 16) {
                _mm_prefetch(&a(xBase + 16, y), _MM_HINT_T0);
                for (int x = xBase; x < xBase + 16; x += 4) {
                    __m128 res = _mm_setzero_ps();
                    for (int t = -nblur; t <= nblur; t++) {
                        res = _mm_add_ps(res, _mm_loadu_ps(&a(x + t, y)));
                    }
                    _mm_stream_ps(&b(x, y), res);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_x_blur_tiled_prefetched_streamed);

BENCHMARK_MAIN();
