#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <algorithm>

void upperIfelse(char *p, int n) {
    for (int i = 0; i < n; i++) {
        if ('a' <= p[i] && p[i] <= 'z')
            p[i] = p[i] + 'A' - 'a';
    }
}

void upperTernary(char *p, int n) {
    for (int i = 0; i < n; i++) {
        p[i] = ('a' <= p[i] && p[i] <= 'z') ? p[i] + 'A' - 'a' : p[i];
    }
}

template <void (*upper)(char *p, int n)>
void testSorted(benchmark::State &state) {
    const int n = (int)1e7;
    std::vector<char> a(n);

    std::mt19937 rng;
    std::uniform_int_distribution<char> dist(0, 127);
    for (int i = 0; i < n; i++) {
        a[i] = dist(rng);
    }
    std::sort(a.begin(), a.end());

    for (auto _: state) {
        upper(a.data(), n);
        benchmark::DoNotOptimize(a);
    }
}

template <void (*upper)(char *p, int n)>
void testRandom(benchmark::State &state) {
    const int n = (int)1e7;
    std::vector<char> a(n);

    std::mt19937 rng;
    std::uniform_int_distribution<char> dist(0, 127);
    for (int i = 0; i < n; i++) {
        a[i] = dist(rng);
    }

    for (auto _: state) {
        upper(a.data(), n);
        benchmark::DoNotOptimize(a);
    }
}

BENCHMARK_TEMPLATE(testSorted, upperTernary);
BENCHMARK_TEMPLATE(testSorted, upperIfelse);
BENCHMARK_TEMPLATE(testRandom, upperTernary);
BENCHMARK_TEMPLATE(testRandom, upperIfelse);

BENCHMARK_MAIN();
/* 
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
testSorted<upperTernary>     547329 ns       546965 ns         1279
testSorted<upperIfelse>     4597908 ns      4597300 ns          153
testRandom<upperTernary>     548555 ns       548232 ns         1296
testRandom<upperIfelse>     7031812 ns      7031073 ns           87
*/
