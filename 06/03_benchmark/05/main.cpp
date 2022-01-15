#include <iostream>
#include <vector>
#include <cmath>
#include <benchmark/benchmark.h>

constexpr size_t n = 1<<27;
std::vector<float> a(n);

void BM_for(benchmark::State &bm) {
    for (auto _: bm) {
        // fill a with sin(i)
        for (size_t i = 0; i < a.size(); i++) {
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_for);

void BM_reduce(benchmark::State &bm) {
    for (auto _: bm) {
        // calculate sum of a
        float res = 0;
        for (size_t i = 0; i < a.size(); i++) {
            res += a[i];
        }
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_reduce);

BENCHMARK_MAIN();
