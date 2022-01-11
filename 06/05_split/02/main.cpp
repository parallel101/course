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
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&] (tbb::blocked_range<size_t> r) {
            mtprint("thread", tbb::this_task_arena::current_thread_index(), "size", r.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }, tbb::auto_partitioner{});
    });

    return 0;
}
