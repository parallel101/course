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

constexpr int nx = 1<<13;
constexpr int ny = 1<<13;

std::vector<float> arr(nx * ny);

void BM_addone(benchmark::State &bm) {
    for (auto _: bm) {
#pragma omp parallel for
        for (int i = 0; i < nx * ny; i++) {
            arr[i] += 1;
        }
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(BM_addone);

void BM_rbgs(benchmark::State &bm) {
    auto a = [&] (int x, int y) -> auto & {
        y = std::clamp(y, 0, ny - 1);
        x = std::clamp(x, 0, nx - 1);
        return arr[y * nx + x];
    };

    for (auto _: bm) {
        for (int phase = 0; phase < 32; phase++) {
#pragma omp parallel for collapse(2)
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    if ((x + y) % 2 == phase % 2)
                        a(x, y) = (a(x - 1, y) + a(x + 1, y) + a(x, y - 1) + a(x, y + 1)) / 4;
                }
            }
        }
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(BM_rbgs);

void BM_rbgs_rearranged(benchmark::State &bm) {
    auto a = [&] (int x, int y) -> auto & {
        y = std::clamp(y, 0, ny - 1);
        x = std::clamp(x, 0, nx - 1);
        return arr[((x + y) % 2) * (nx / 2 * ny) + ((y / 2) * nx + x)];
    };

    for (auto _: bm) {
        for (int phase = 0; phase < 32; phase++) {
#pragma omp parallel for collapse(2)
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    if ((x + y) % 2 == phase % 2)
                        a(x, y) = (a(x - 1, y) + a(x + 1, y) + a(x, y - 1) + a(x, y + 1)) / 4;
                }
            }
        }
        benchmark::DoNotOptimize(arr);
    }
}
BENCHMARK(BM_rbgs_rearranged);

BENCHMARK_MAIN();
