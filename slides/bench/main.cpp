#include <benchmark/benchmark.h>

void BM_latency(benchmark::State &_) {
    void *tmp;
    tmp = &tmp;
    register void *rax asm ("rax");
    rax = tmp;
    for (auto _ : _) {
        const size_t repeats = 10000;
        #pragma GCC unroll 100
        for (size_t i = 0; i < repeats; ++i) {
            rax = *(void **)rax;
        }
    }
    tmp = rax;
    benchmark::DoNotOptimize(tmp);
}
BENCHMARK(BM_latency)->MinWarmUpTime(0.2)->MinTime(1.5);

void BM_latency_aligned(benchmark::State &_) {
    alignas(void *) char buf[sizeof(void *) * 2];
    void *&tmp = *new (buf) void *;
    tmp = &tmp;
    register void *rax asm ("rax");
    rax = tmp;
    for (auto _ : _) {
        const size_t repeats = 10000;
        #pragma GCC unroll 100
        for (size_t i = 0; i < repeats; ++i) {
            rax = *(void **)rax;
        }
    }
    tmp = rax;
    benchmark::DoNotOptimize(tmp);
}
BENCHMARK(BM_latency_aligned)->MinWarmUpTime(0.2)->MinTime(1.5);

void BM_latency_unaligned(benchmark::State &_) {
    alignas(void *) char buf[sizeof(void *) * 2];
    void *&tmp = *new (buf + 1) void *;
    tmp = &tmp;
    register void *rax asm ("rax");
    rax = tmp;
    for (auto _ : _) {
        const size_t repeats = 10000;
        #pragma GCC unroll 100
        for (size_t i = 0; i < repeats; ++i) {
            rax = *(void **)rax;
        }
    }
    tmp = rax;
    benchmark::DoNotOptimize(tmp);
}
BENCHMARK(BM_latency_unaligned)->MinWarmUpTime(0.2)->MinTime(1.5);

void BM_throughput(benchmark::State &_) {
    const size_t size = 10000;
    void **arr = new void *[size];
    for (size_t i = 0; i < size; ++i) {
        arr[i] = &arr[i];
    }
    for (auto _ : _) {
        #pragma GCC unroll 100
        for (void **p = arr, **end = arr + size; p < end; ++p) {
            *p = *(void **)*p;
        }
    }
    benchmark::DoNotOptimize(arr);
    delete[] arr;
}
BENCHMARK(BM_throughput)->MinWarmUpTime(0.2)->MinTime(1.5);

void BM_throughput_aligned(benchmark::State &_) {
    const size_t size = 10000;
    void **arr = new void *[size + 1];
    for (size_t i = 0; i < size; ++i) {
        arr[i] = &arr[i];
    }
    for (auto _ : _) {
        #pragma GCC unroll 100
        for (void **p = arr, **end = arr + size; p < end; ++p) {
            *p = *(void **)*p;
        }
    }
    benchmark::DoNotOptimize(arr);
    delete[] arr;
}
BENCHMARK(BM_throughput_aligned)->MinWarmUpTime(0.2)->MinTime(1.5);

void BM_throughput_unaligned(benchmark::State &_) {
    const size_t size = 10000;
    void **arr0 = new void *[size + 1];
    void **arr = (void **)((char *)arr0 + 1);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = &arr[i];
    }
    for (auto _ : _) {
        #pragma GCC unroll 100
        for (void **p = arr, **end = arr + size; p < end; ++p) {
            *p = *(void **)*p;
        }
    }
    benchmark::DoNotOptimize(arr);
    delete[] arr0;
}
BENCHMARK(BM_throughput_unaligned)->MinWarmUpTime(0.2)->MinTime(1.5);
