#include <iostream>
#include <tbb/task_group.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    size_t maxt = 4;
    tbb::task_group tg;
    for (size_t t = 0; t < maxt; t++) {
        auto beg = t * n / maxt;
        auto end = std::min(n, (t + 1) * n / maxt);
        tg.run([&, beg, end] {
            for (size_t i = beg; i < end; i++) {
                a[i] = std::sin(i);
            }
        });
    }
    tg.wait();

    return 0;
}
