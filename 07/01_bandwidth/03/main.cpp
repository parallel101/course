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

static float func(float x) {
    return x * (x * x + x * 3.14f - 1 / (x + 1)) + 42 / (2.718f - x);
}

void BM_serial_func(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; i++) {
            a[i] = func(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_serial_func);

void BM_parallel_func(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = func(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_parallel_func);

BENCHMARK_MAIN();
