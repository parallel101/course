#include <iostream>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <cmath>
#include "ticktock.h"

int main() {
    size_t n = 1<<27;

    TICK(filter);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), std::vector<float>{},
    [&] (tbb::blocked_range<size_t> r, std::vector<float> local_a) {
        local_a.reserve(local_a.size() + r.size());
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                local_a.push_back(val);
            }
        }
        return local_a;
    }, [] (std::vector<float> a, std::vector<float> const &b) {
        std::copy(b.begin(), b.end(), std::back_inserter(a));
        return a;
    });
    TOCK(filter);

    return 0;
}
