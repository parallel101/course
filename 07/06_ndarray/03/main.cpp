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

void BM_x_blur(benchmark::State &bm) {
    constexpr int nblur = 8;
    ndarray<2, float, nblur> a(nx, ny);
    ndarray<2, float> b(nx, ny);

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

void BM_stdvector(benchmark::State &bm) {
    std::vector<float> a(nx * ny);

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a[y * nx + x] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_stdvector);

void BM_ndarray(benchmark::State &bm) {
    ndarray<2, float> a(nx, ny);

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a(x, y) = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ndarray);

void BM_ndarray_aligned(benchmark::State &bm) {
    ndarray<2, float> a(nx, ny);

    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x += 8) {
                _mm256_stream_ps(&a(x, y), _mm256_set1_ps(1));
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ndarray_aligned);

BENCHMARK_MAIN();
