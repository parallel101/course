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

ndarray<2, float, nblur, nblur, 32> a(nx, ny);
ndarray<2, float, 0, 0, 32> b(nx, ny);

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

void BM_y_blur(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float res = 0;
                for (int t = -nblur; t <= nblur; t++) {
                    res += a(x, y + t);
                }
                b(x, y) = res;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur);

void BM_y_blur_tiled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int yBase = 0; yBase < ny; yBase += blockSize) {
            for (int xBase = 0; xBase < nx; xBase += blockSize) {
                for (int y = yBase; y < yBase + blockSize; y++) {
                    for (int x = xBase; x < xBase + blockSize; x++) {
                        float res = 0;
                        for (int t = -nblur; t <= nblur; t++) {
                            res += a(x, y + t);
                        }
                        b(x, y) = res;
                    }
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled);

void BM_y_blur_tiled_only_x(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int x = xBase; x < xBase + blockSize; x++) {
                    float res = 0;
                    for (int t = -nblur; t <= nblur; t++) {
                        res += a(x, y + t);
                    }
                    b(x, y) = res;
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x);

void BM_y_blur_tiled_only_x_prefetched_streamed(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 1024;
#pragma omp parallel for collapse(2)
        for (int xBase = 0; xBase < nx; xBase += blockSize) {
            for (int y = 0; y < ny; y++) {
                for (int x = xBase; x < xBase + blockSize; x += 32) {
                    _mm_prefetch(&a(x + 0, y + nblur), _MM_HINT_T0);
                    _mm_prefetch(&a(x + 16, y + nblur), _MM_HINT_T0);
                    __m256 res0 = _mm256_setzero_ps();
                    __m256 res1 = _mm256_setzero_ps();
                    __m256 res2 = _mm256_setzero_ps();
                    __m256 res3 = _mm256_setzero_ps();
                    for (int t = -nblur; t <= nblur; t++) {
                        res0 = _mm256_add_ps(res0, _mm256_load_ps(&a(x + 0, y + t)));
                        res1 = _mm256_add_ps(res1, _mm256_load_ps(&a(x + 8, y + t)));
                        res2 = _mm256_add_ps(res2, _mm256_load_ps(&a(x + 16, y + t)));
                        res3 = _mm256_add_ps(res3, _mm256_load_ps(&a(x + 24, y + t)));
                    }
                    _mm256_stream_ps(&b(x + 0, y), res0);
                    _mm256_stream_ps(&b(x + 8, y), res1);
                    _mm256_stream_ps(&b(x + 16, y), res2);
                    _mm256_stream_ps(&b(x + 24, y), res3);
                }
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_y_blur_tiled_only_x_prefetched_streamed);

BENCHMARK_MAIN();
