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

constexpr int n = 1<<23;

std::vector<float> a(n);

void BM_false_sharing(benchmark::State &bm) {
    for (auto _: bm) {
        std::vector<int> tmp(omp_get_max_threads());
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            tmp[omp_get_thread_num()] += a[i];
            benchmark::DoNotOptimize(tmp);
        }
        benchmark::DoNotOptimize(tmp);
    }
}
BENCHMARK(BM_false_sharing);

void BM_no_false_sharing(benchmark::State &bm) {
    for (auto _: bm) {
        std::vector<int> tmp(omp_get_max_threads() * 4096);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            tmp[omp_get_thread_num() * 4096] += a[i];
            benchmark::DoNotOptimize(tmp);
        }
        benchmark::DoNotOptimize(tmp);
    }
}
BENCHMARK(BM_no_false_sharing);

BENCHMARK_MAIN();
