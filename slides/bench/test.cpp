#include <benchmark/benchmark.h>

void BM_latency(benchmark::State &state) {
    size_t offset = state.range(0);
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
    benchmark::DoNotOptimize(rax);
}
BENCHMARK(BM_latency)->MinTime(0.05)->ArgName("offset")->DenseRange(0, 80);

struct alignas(8) A {
    uint32_t a;
    uint32_t b;
};

void BM_struct(benchmark::State &state) {
    size_t offset = state.range(0);
    const size_t repeats = 100;
    alignas(64) static char buf[256];
    A &tmp = *new (buf + offset) A;
    tmp.a = 0;
    tmp.b = 0;
    benchmark::DoNotOptimize(tmp);
    uintptr_t rax = (uintptr_t)&tmp;
    for (auto _: state) {
        #pragma GCC unroll repeats
        for (size_t i = 0; i < repeats; ++i) {
            // movzx rbx, dword [rax]; add rax, rbx; movzx rbx, dword [rax + 4]; add rax, rbx
            rax += ((A *)rax)->a;
            rax += ((A *)rax)->b;
        }
    }
    benchmark::DoNotOptimize(rax);
}
BENCHMARK(BM_struct)->MinTime(0.05)->ArgName("offset")->DenseRange(0, 80);
