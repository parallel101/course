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
#ifdef WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#endif

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr int n = 1<<10;
constexpr int nblur = 8;

ndarray<2, float, 16> a(n, n);
ndarray<2, float> b(n, n);

void BM_y_blur(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                for (int t = 0; t < nblur; t++) {
                    b(x, y) += a(x, y + t);
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_y_blur);

void BM_y_blur_tiled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for collapse(2)
        for (int yBase = 0; yBase < n; yBase += blockSize) {
            for (int xBase = 0; xBase < n; xBase += blockSize) {
                for (int y = yBase; y < yBase + blockSize; y++) {
                    for (int x = xBase; x < xBase + blockSize; x++) {
                        for (int t = -nblur; t <= nblur; t++) {
                            b(x, y) += a(x, y + t);
                        }
                    }
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_y_blur_tiled);

void BM_y_blur_morton_tiled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 32;
#pragma omp parallel for
        for (int mortonCode = 0; mortonCode < n * n / blockSize / blockSize; mortonCode++) {
            auto [xBase, yBase] = morton2d::decode(mortonCode);
            xBase *= blockSize;
            yBase *= blockSize;
            for (int y = yBase; y < yBase + blockSize; y++) {
                for (int x = xBase; x < xBase + blockSize; x++) {
                    for (int t = -nblur; t <= nblur; t++) {
                        b(x, y) += a(x, y + t);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_y_blur_morton_tiled);

BENCHMARK_MAIN();
