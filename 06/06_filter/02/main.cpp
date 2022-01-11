#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <cmath>
#include "ticktock.h"

int main() {
    size_t n = 1<<27;
    tbb::concurrent_vector<float> a;

    TICK(filter);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                a.push_back(val);
            }
        }
    });
    TOCK(filter);

    return 0;
}
