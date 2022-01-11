#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <cmath>
#include "ticktock.h"
#include "pod.h"

int main() {
    size_t n = 1<<27;

    TICK(filter);

    TICK(init);
    std::vector<pod<float>> a(n);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            a[i] = std::sin(i);
        }
    });
    TOCK(init);

    TICK(scan);
    std::vector<pod<size_t>> ind(n + 1);
    ind[0] = 0;
    tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), (size_t)0,
    [&] (tbb::blocked_range<size_t> r, size_t sum, auto is_final) {
        for (auto i = r.begin(); i < r.end(); i++) {
            sum += a[i] > 0 ? 1 : 0;
            if (is_final)
                ind[i + 1] = sum;
        }
        return sum;
    }, [] (size_t x, size_t y) {
        return x + y;
    });
    TOCK(scan);

    TICK(fill);
    std::vector<pod<float>> b(ind.back());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            if (a[i] > 0)
                b[ind[i]] = a[i];
        }
    });
    TOCK(fill);

    TOCK(filter);

    return 0;
}
