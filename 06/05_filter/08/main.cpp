#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>
#include <atomic>
#include "ticktock.h"
#include "pod.h"

int main() {
    size_t n = 1<<27;
    std::vector<pod<float>> a;
    std::atomic<size_t> a_size = 0;

    TICK(filter);
    a.resize(n);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<pod<float>> local_a(r.size());
        size_t lasize = 0;
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                local_a[lasize++] = val;
            }
        }
        size_t base = a_size.fetch_add(lasize);
        for (size_t i = 0; i < lasize; i++) {
            a[base + i] = local_a[i];
        }
    });
    a.resize(a_size);
    TOCK(filter);

    return 0;
}
