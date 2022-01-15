#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <x86intrin.h>
#include <omp.h>

void BM_fill(benchmark::State &bm) {
    for (auto _: bm) {
    }
}
BENCHMARK(BM_fill);

BENCHMARK_MAIN();
