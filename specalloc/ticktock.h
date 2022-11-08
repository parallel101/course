#pragma once

#include <chrono>
#include <cstdio>

#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x) std::printf("%s: %lfs\n", #x, std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x).count());

template <class T>
#ifdef _MSC_VER
__noinline
#else
__attribute__((noinline))
#endif
static void NOPT(T const &t) {
    (void)t;
}
