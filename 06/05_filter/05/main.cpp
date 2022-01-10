#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>
#include <mutex>
#include "ticktock.h"

int main() {
    size_t n = 1<<27;
    std::vector<float> a;
    std::mutex mtx;

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
        std::lock_guard lck(mtx);
        std::copy(local_a.begin(), local_a.end(), std::back_inserter(a));
    });
    TOCK(filter);

    return 0;
}
