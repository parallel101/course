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

constexpr size_t nx = 1<<13;
constexpr size_t ny = 1<<13;

ndarray<2, float> a(nx, ny);
ndarray<2, float> b(nx, ny);

void BM_copy(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                b(x, y) = a(x, y);
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_copy);

void BM_copy_streamed(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x += 4) {
                _mm_stream_ps(&b(x, y), _mm_load_ps(&a(x, y)));
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_copy_streamed);

void BM_transpose(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                b(x, y) = a(y, x);
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_transpose);

void BM_transpose_tiled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 64;  // 16KB
#pragma omp parallel for collapse(2)
        for (int yBase = 0; yBase < ny; yBase += blockSize) {
            for (int xBase = 0; xBase < nx; xBase += blockSize) {
                for (int y = yBase; y < yBase + blockSize; y++) {
                    for (int x = xBase; x < xBase + blockSize; x++) {
                        b(x, y) = a(y, x);
                    }
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_transpose_tiled);

void BM_transpose_morton_tiled(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 64;  // 16KB
#pragma omp parallel for
        for (int mortonCode = 0; mortonCode < (ny / blockSize) * (nx / blockSize); mortonCode++) {
            auto [xBase, yBase] = morton2d::decode(mortonCode);
            xBase *= blockSize;
            yBase *= blockSize;
            for (int y = yBase; y < yBase + blockSize; y++) {
                for (int x = xBase; x < xBase + blockSize; x++) {
                    b(x, y) = a(y, x);
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_transpose_morton_tiled);

void BM_transpose_morton_tiled_streamed(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 64;  // 16KB
#pragma omp parallel for
        for (int mortonCode = 0; mortonCode < (ny / blockSize) * (nx / blockSize); mortonCode++) {
            auto [xBase, yBase] = morton2d::decode(mortonCode);
            xBase *= blockSize;
            yBase *= blockSize;
            for (int y = yBase; y < yBase + blockSize; y++) {
                for (int x = xBase; x < xBase + blockSize; x += 4) {
                    _mm_stream_si32((int *)&b(x, y), (int &)a(y, x));
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_transpose_morton_tiled_streamed);

void BM_transpose_morton_tiled_streamed_reversed(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 64;  // 16KB
#pragma omp parallel for
        for (int mortonCode = 0; mortonCode < (ny / blockSize) * (nx / blockSize); mortonCode++) {
            auto [yBase, xBase] = morton2d::decode(mortonCode);
            xBase *= blockSize;
            yBase *= blockSize;
            for (int y = yBase; y < yBase + blockSize; y++) {
                for (int x = xBase; x < xBase + blockSize; x += 4) {
                    _mm_stream_si32((int *)&b(x, y), (int &)a(y, x));
                }
            }
        }
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_transpose_morton_tiled_streamed_reversed);

#ifdef WITH_TBB
void BM_transpose_tbb(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 64;  // 16KB
        tbb::parallel_for(tbb::blocked_range2d<size_t>(0, nx, blockSize, 0, ny, blockSize),
        [&] (tbb::blocked_range2d<size_t> const &r) {
            for (int y = r.cols().begin(); y < r.cols().end(); y++) {
                for (int x = r.rows().begin(); x < r.rows().end(); x++) {
                    b(x, y) = a(y, x);
                }
            }
        }, tbb::simple_partitioner{});
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_transpose_tbb);

void BM_transpose_tbb_reversed(benchmark::State &bm) {
    for (auto _: bm) {
        constexpr int blockSize = 64;  // 16KB
        tbb::parallel_for(tbb::blocked_range2d<size_t>(0, ny, blockSize, 0, nx, blockSize),
        [&] (tbb::blocked_range2d<size_t> const &r) {
            for (int y = r.rows().begin(); y < r.rows().end(); y++) {
                for (int x = r.cols().begin(); x < r.cols().end(); x++) {
                    b(x, y) = a(y, x);
                }
            }
        }, tbb::simple_partitioner{});
        benchmark::DoNotOptimize(b);
    }
}
BENCHMARK(BM_transpose_tbb_reversed);
#endif

BENCHMARK_MAIN();
