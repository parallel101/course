#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<13;
    std::vector<float> a(n * n);

    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, n, 0, n),
    [&] (tbb::blocked_range2d<size_t> r) {
        for (size_t i = r.cols().begin(); i < r.cols().end(); i++) {
            for (size_t j = r.rows().begin(); j < r.rows().end(); j++) {
                a[i * n + j] = std::sin(i) * std::sin(j);
            }
        }
    });

    return 0;
}
