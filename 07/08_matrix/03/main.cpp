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

constexpr int n = 1<<9;
constexpr int nkern = 16;

ndarray<2, float> a(n, n);
ndarray<2, float, nkern> b(n, n);
ndarray<2, float> c(nkern, nkern);

void BM_convol(benchmark::State &bm) {
    for (auto _: bm) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                for (int l = 0; l < nkern; l++) {
                    for (int k = 0; k < nkern; k++) {
                        a(i, j) += b(i + k, j + l) * c(i, j);
                    }
                }
            }
        }
    }
}
BENCHMARK(BM_convol);

void BM_convol_blocked(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 4;
        for (int jBase = 0; jBase < n; jBase += blockSize) {
            for (int iBase = 0; iBase < n; iBase += blockSize) {
                for (int l = 0; l < nkern; l++) {
                    for (int k = 0; k < nkern; k++) {
                        for (int j = jBase; j < jBase + blockSize; j++) {
                            for (int i = iBase; i < iBase + blockSize; i++) {
                                a(i, j) += b(i + k, j + l) * c(i, j);
                            }
                        }
                    }
                }
            }
        }
    }
}
BENCHMARK(BM_convol_blocked);

void BM_convol_blocked_unrolled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 4;
        for (int jBase = 0; jBase < n; jBase += blockSize) {
            for (int iBase = 0; iBase < n; iBase += blockSize) {
                for (int l = 0; l < nkern; l++) {
                    for (int k = 0; k < nkern; k++) {
                        for (int j = jBase; j < jBase + blockSize; j++) {
#pragma GCC unroll 4
                            for (int i = iBase; i < iBase + blockSize; i++) {
                                a(i, j) += b(i + k, j + l) * c(i, j);
                            }
                        }
                    }
                }
            }
        }
    }
}
BENCHMARK(BM_convol_blocked_unrolled);

BENCHMARK_MAIN();
