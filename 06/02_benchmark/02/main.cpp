#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>
#include "ticktock.h"

int main() {
    size_t n = 1<<27;
    std::vector<float> a(n);

    TICK(for);
    // fill a with sin(i)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            a[i] = std::sin(i);
        }
    });
    TOCK(for);

    TICK(reduce);
    // calculate sum of a
    float res = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), (float)0,
    [&] (tbb::blocked_range<size_t> r, float local_res) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += a[i];
        }
        return local_res;
    }, [] (float x, float y) {
        return x + y;
    });
    TOCK(reduce);

    std::cout << res << std::endl;
    return 0;
}
