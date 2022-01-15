#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>

constexpr size_t n = 1<<28;

std::vector<float> a(n);  // 1GB

static float funcA(float x) {
    return sqrtf(x) * x;
}

void BM_1funcA(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(1);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcA(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_1funcA);

void BM_2funcA(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(2);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcA(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_2funcA);

void BM_4funcA(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(4);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcA(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_4funcA);

void BM_6funcA(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(6);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcA(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_6funcA);

void BM_8funcA(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(8);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcA(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_8funcA);

void BM_10funcA(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(10);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcA(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_10funcA);

BENCHMARK_MAIN();
