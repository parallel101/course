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
#include "morton.h"
#ifdef WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#endif

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr size_t nx = 1<<13;
constexpr size_t ny = 1<<13;

ndarray<2, float> a(nx, ny);

void BM_addone(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a(x, y) += 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_addone);

void BM_rbgs(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a(x, y) = (a(x - 1, y) + a(x + 1, y) + a(x, y - 1) + a(x, y + 1)) / 4;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_rbgs);

BENCHMARK_MAIN();
