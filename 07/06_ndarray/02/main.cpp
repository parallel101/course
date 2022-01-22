#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr size_t nx = 1<<13;
constexpr size_t ny = 1<<13;

std::vector<float> a(nx * ny);

void BM_yx_loop_yx_array(benchmark::State &bm) {
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
BENCHMARK(BM_yx_loop_yx_array);

void BM_xy_loop_yx_array(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                a[y * nx + x] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_xy_loop_yx_array);

void BM_yx_loop_xy_array(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a[y + x * ny] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_yx_loop_xy_array);

void BM_xy_loop_xy_array(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                a[y + x * ny] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_xy_loop_xy_array);

BENCHMARK_MAIN();
