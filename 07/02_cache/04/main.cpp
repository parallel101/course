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

constexpr size_t n = 1<<26;  // 512MB

void BM_aos(benchmark::State &bm) {
    struct MyClass {
        float x;
        float y;
        float z;
    };

    std::vector<MyClass> mc(n);

    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            mc[i].x += 1;
            mc[i].y += 1;
            mc[i].z += 1;
        }
        benchmark::DoNotOptimize(mc);
    }
}
BENCHMARK(BM_aos);

void BM_soa(benchmark::State &bm) {
    std::vector<float> mc_x(n);
    std::vector<float> mc_y(n);
    std::vector<float> mc_z(n);

    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
                mc_x[i] += 1;
                mc_y[i] += 1;
                mc_z[i] += 1;
        }
        benchmark::DoNotOptimize(mc_x);
        benchmark::DoNotOptimize(mc_y);
        benchmark::DoNotOptimize(mc_z);
    }
}
BENCHMARK(BM_soa);

void BM_aosoa(benchmark::State &bm) {
    struct MyClass {
        float x[1024];
        float y[1024];
        float z[1024];
    };

    std::vector<MyClass> mc(n / 1024);

    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 1024; i++) {
#pragma omp simd
            for (size_t j = 0; j < 1024; j++) {
                mc[i].x[j] += 1;
                mc[i].y[j] += 1;
                mc[i].z[j] += 1;
            }
        }
        benchmark::DoNotOptimize(mc);
    }
}
BENCHMARK(BM_aosoa);

BENCHMARK_MAIN();
