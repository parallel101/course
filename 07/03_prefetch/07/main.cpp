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

constexpr size_t n = 1<<27;  // 512MB

std::vector<int> a(n);

void BM_write0(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = 0;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write0);

void BM_write1(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write1);

void BM_write1_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            _mm_stream_si32(&a[i], 1);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_write1_streamed);

BENCHMARK_MAIN();
