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

BENCHMARK_MAIN();
