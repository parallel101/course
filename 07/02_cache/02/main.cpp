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

constexpr size_t n = 1<<28;
std::vector<float> a(n);  // 1GB

void BM_skip1(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 1) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip1);

void BM_skip2(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 2) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip2);

void BM_skip4(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 4) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip4);

void BM_skip8(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 8) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip8);

void BM_skip16(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 16) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip16);

void BM_skip32(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 32) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip32);

void BM_skip64(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 64) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip64);

void BM_skip128(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i += 128) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip128);

BENCHMARK_MAIN();
