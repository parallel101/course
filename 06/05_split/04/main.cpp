#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>
#include <thread>
#include "ticktock.h"
#include "mtprint.h"

int main() {
    size_t n = 32;

    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::affinity_partitioner affinity;
        for (int t = 0; t < 10; t++) {
            TICK(for);
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
            [&] (tbb::blocked_range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    for (volatile int j = 0; j < i * 1000; j++);
                }
            }, affinity);
            TOCK(for);
        }
    });

    return 0;
}
