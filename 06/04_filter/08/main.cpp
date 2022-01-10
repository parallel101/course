#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>
#include <atomic>
#include "ticktock.h"

int main() {
    size_t n = 1<<27;
    std::vector<float> a(n);
    std::atomic<size_t> a_size = 0;

    TICK(filter);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<float> local_a;
        local_a.reserve(r.size());
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                local_a.push_back(val);
            }
        }
        size_t base = a_size.fetch_add(local_a.size());
        for (size_t i = 0; i < local_a.size(); i++) {
            a[base + i] = local_a[i];
        }
    });
    a.resize(a_size);
    TOCK(filter);

    return 0;
}
