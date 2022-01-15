#include <iostream>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);
    for (size_t i = 0; i < n; i++) {
        a[i] = 10.f + std::sin(i);
    }

    float serial_avg = 0;
    for (size_t i = 0; i < n; i++) {
        serial_avg += a[i];
    }
    serial_avg /= n;
    std::cout << serial_avg << std::endl;

    float parallel_avg = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), (float)0,
    [&] (tbb::blocked_range<size_t> r, float local_avg) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_avg += a[i];
        }
        return local_avg;
    }, [] (float x, float y) {
        return x + y;
    }) / n;

    std::cout << parallel_avg << std::endl;
    return 0;
}
