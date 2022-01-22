#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>
#include "ndarray.h"
#include "morton.h"

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr int n = 1<<10;

ndarray<2, float> a(n, n);
ndarray<2, float> b(n, n);
ndarray<2, float> c(n, n);

void BM_matmul(benchmark::State &bm) {
    for (auto _: bm) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                for (int t = 0; t < n; t++) {
                    a(i, j) += b(i, t) * c(t, j);
                }
            }
        }
    }
}
BENCHMARK(BM_matmul);

void BM_matmul_blocked(benchmark::State &bm) {
    for (auto _: bm) {
        for (int j = 0; j < n; j++) {
            for (int iBase = 0; iBase < n; iBase += 32) {
                for (int t = 0; t < n; t++) {
                    for (int i = iBase; i < iBase + 32; i++) {
                        a(i, j) += b(i, t) * c(t, j);
                    }
                }
            }
        }
    }
}
BENCHMARK(BM_matmul_blocked);

void BM_matmul_blocked_both(benchmark::State &bm) {
    for (auto _: bm) {
        for (int jBase = 0; jBase < n; jBase += 16) {
            for (int iBase = 0; iBase < n; iBase += 16) {
                for (int j = jBase; j < jBase + 16; j++) {
                    for (int t = 0; t < n; t++) {
                        for (int i = iBase; i < iBase + 16; i++) {
                            a(i, j) += b(i, t) * c(t, j);
                        }
                    }
                }
            }
        }
    }
}
BENCHMARK(BM_matmul_blocked_both);

BENCHMARK_MAIN();
