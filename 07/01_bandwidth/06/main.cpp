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

static float funcC(float x) {
    return sinf(x) + cosf((x + 1) * x) * sqrtf(x * (x + 1));
}

void BM_1funcC(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(1);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcC(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_1funcC);

void BM_2funcC(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(2);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcC(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_2funcC);

void BM_4funcC(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(4);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcC(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_4funcC);

void BM_6funcC(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(6);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcC(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_6funcC);

void BM_8funcC(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(8);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcC(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_8funcC);

void BM_10funcC(benchmark::State &bm) {
    for (auto _: bm) {
        omp_set_num_threads(10);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = funcC(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_10funcC);

BENCHMARK_MAIN();
