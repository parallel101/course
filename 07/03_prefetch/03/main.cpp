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

std::vector<float> a(n);

static uint32_t randomize(uint32_t i) {
	i = (i ^ 61) ^ (i >> 16);
	i *= 9;
	i ^= i << 4;
	i *= 0x27d4eb2d;
	i ^= i >> 15;
    return i;
}

void BM_ordered(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            benchmark::DoNotOptimize(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ordered);

void BM_random_64B(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 16; i++) {
            size_t r = randomize(i) % (n / 16);
            for (size_t j = 0; j < 16; j++) {
                benchmark::DoNotOptimize(a[r * 16 + j]);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_64B);

void BM_random_4KB(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 1024; i++) {
            size_t r = randomize(i) % (n / 1024);
            for (size_t j = 0; j < 1024; j++) {
                benchmark::DoNotOptimize(a[r * 1024 + j]);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_4KB);

void BM_random_4KB_aligned(benchmark::State &bm) {
    float *a = (float *)_mm_malloc(n * sizeof(float), 4096);
    memset(a, 0, n * sizeof(float));
    for (auto _: bm) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 1024; i++) {
            size_t r = randomize(i) % (n / 1024);
            for (size_t j = 0; j < 1024; j++) {
                benchmark::DoNotOptimize(a[r * 1024 + j]);
            }
        }
        benchmark::DoNotOptimize(a);
    }
    _mm_free(a);
}
BENCHMARK(BM_random_4KB_aligned);

BENCHMARK_MAIN();
