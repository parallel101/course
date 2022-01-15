#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>

constexpr size_t n = 1<<26;

std::vector<float> a(n);  // 256MB

void BM_fill(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill);

void BM_parallel_fill(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_parallel_fill);

void BM_sine(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < n; i++) {
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_sine);

void BM_parallel_sine(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_parallel_sine);

BENCHMARK_MAIN();
