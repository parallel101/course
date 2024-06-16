#include <benchmark/benchmark.h>

void BM_latency(benchmark::State &state) {
    int offset = state.range(0);
    const size_t repeats = 100;
    alignas(64) static char buf[256];
    uintptr_t &tmp = *new (buf + offset) uintptr_t;
    tmp = (uintptr_t)&tmp;
    uintptr_t rax = tmp;
    for (auto _: state) {
        #pragma GCC unroll repeats
        for (size_t i = 0; i < repeats; ++i) {
            rax = *(uintptr_t *)rax; // mov rax, [rax]
        }
    }
    tmp = rax;
    benchmark::DoNotOptimize(tmp);
}
BENCHMARK(BM_latency)->MinTime(0.05)->ArgName("offset")->DenseRange(0, 80);
