#include <iostream>
#include <tbb/task_group.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    float res = 0;

    size_t maxt = 4;
    tbb::task_group tg;
    std::vector<float> tmp_res(maxt);
    for (size_t t = 0; t < maxt; t++) {
        size_t beg = t * n / maxt;
        size_t end = std::min(n, (t + 1) * n / maxt);
        tg.run([&, t, beg, end] {
            float local_res = 0;
            for (size_t i = beg; i < end; i++) {
                local_res += std::sin(i);
            }
            tmp_res[t] = local_res;
        });
    }
    tg.wait();
    for (size_t t = 0; t < maxt; t++) {
        res += tmp_res[t];
    }

    std::cout << res << std::endl;
    return 0;
}
