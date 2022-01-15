#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <array>

constexpr size_t n = 1<<29;
//int *a = new int[n]{};
int *a = (int *)memset(_mm_malloc(n * sizeof(int), 4096), 0, n * sizeof(int));
int *b = (int *)memset(_mm_malloc(n * sizeof(int), 4096), 0, n * sizeof(int));

void BM_simd_copy(benchmark::State &bm) {
    for (auto _: bm) {
        for (int i = 0; i < n; i += 4) {
            _mm_stream_ps((float*)&b[i], _mm_load_ps((float*)&a[i]));
        }
    }
}
BENCHMARK(BM_simd_copy);

void BM_memcpy(benchmark::State &bm) {
    for (auto _: bm) {
        std::memcpy(b, a, sizeof(int) * n);
    }
}
BENCHMARK(BM_memcpy);

void BM_copy(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            b[i] = a[i];
        }
    }
}
BENCHMARK(BM_copy);

void BM_stream_copy(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            _mm_stream_si32(&b[i], a[i]);
        }
    }
}
BENCHMARK(BM_stream_copy);

void BM_read(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            ((volatile int *)a)[i];
        }
    }
}
BENCHMARK(BM_read);

void BM_readwrite(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = a[i] + 1;
        }
    }
}
BENCHMARK(BM_readwrite);

void BM_write1(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = 1;
        }
    }
}
BENCHMARK(BM_write1);

void BM_write0(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = 0;
        }
    }
}
BENCHMARK(BM_write0);

BENCHMARK_MAIN();
