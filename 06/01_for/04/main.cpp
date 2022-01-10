#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    tbb::parallel_for((size_t)0, (size_t)n, [&] (size_t i) {
        a[i] = std::sin(i);
    });

    return 0;
}
