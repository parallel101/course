#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<13;
    std::vector<float> a(n * n);

    tbb::parallel_for((size_t)0, (size_t)n, [&] (size_t i) {
        tbb::parallel_for((size_t)0, (size_t)n, [&] (size_t j) {
            a[i * n + j] = std::sin(i) * std::sin(j);
        });
    });

    return 0;
}
