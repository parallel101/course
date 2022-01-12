#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>
#include <thread>
#include "ticktock.h"
#include "mtprint.h"

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    TICK(for);
    tbb::task_arena ta(4);
    ta.execute([&] {
        auto numprocs = tbb::this_task_arena::max_concurrency();  // 4
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, n / (2 * numprocs)),
        [&] (tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                a[i] = std::sin(i);
            }
        }, tbb::simple_partitioner{});
    });
    TOCK(for);

    return 0;
}
