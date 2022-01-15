#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>

// 32KB, 256KB, 12MB

constexpr size_t n = 1<<28;

std::vector<float> a(n);

void BM_fill1GB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1<<28); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill1GB);

void BM_fill32MB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1<<23); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill32MB);

void BM_fill1MB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1<<18); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill1MB);

void BM_fill128KB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1<<15); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill128KB);

void BM_fill4KB(benchmark::State &bm) {
    for (auto _: bm) {
        for (size_t i = 0; i < (1<<10); i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_fill4KB);

BENCHMARK_MAIN();
