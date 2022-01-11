#include <iostream>
#include <tbb/parallel_for.h>
#include <vector>
#include <cmath>
#include <thread>
#include "ticktock.h"
#include "mtprint.h"

int main() {
    size_t n = 32;

    TICK(for);
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&] (tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                mtprint("thread", tbb::this_task_arena::current_thread_index(),
                        "size", r.size(), "begin", r.begin());
                std::this_thread::sleep_for(std::chrono::milliseconds(i * 10));
            }
        }, tbb::simple_partitioner{});
    });
    TOCK(for);

    return 0;
}
