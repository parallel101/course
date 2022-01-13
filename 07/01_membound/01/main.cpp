#include <iostream>
#include <vector>
#include <cmath>
#include <benchmark/benchmark.h>
#include <omp.h>

const size_t n = 1<<29;
std::vector<float> a(n);

void BM_serial_add(benchmark::State &bm) {
    for (auto _: bm) {
        for (int i = 0; i < n; i++) {
            a[i] = a[i] + 1;
        }
    }
}
BENCHMARK(BM_serial_add);

void BM_parallel_add(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = a[i] + 1;
        }
    }
}
BENCHMARK(BM_parallel_add);

void BM_serial_mul(benchmark::State &bm) {
    for (auto _: bm) {
        for (int i = 0; i < n; i++) {
            a[i] = a[i] * 3.14f;
        }
    }
}
BENCHMARK(BM_serial_mul);

void BM_parallel_mul(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = a[i] * 3.14f;
        }
    }
}
BENCHMARK(BM_parallel_mul);

void BM_serial_div(benchmark::State &bm) {
    for (auto _: bm) {
        for (int i = 0; i < n; i++) {
            a[i] = 3.14f / a[i];
        }
    }
}
BENCHMARK(BM_serial_div);

void BM_parallel_div(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = 3.14f / a[i];
        }
    }
}
BENCHMARK(BM_parallel_div);

void BM_serial_sqrt(benchmark::State &bm) {
    for (auto _: bm) {
        for (int i = 0; i < n; i++) {
            a[i] = std::sqrt(a[i]);
        }
    }
}
BENCHMARK(BM_serial_sqrt);

void BM_parallel_sqrt(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = std::sqrt(a[i]);
        }
    }
}
BENCHMARK(BM_parallel_sqrt);

BENCHMARK_MAIN();
