#include <iostream>
#include <tbb/task_group.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);
    float res = 0;

    size_t maxt = 4;
    tbb::task_group tg1;
    std::vector<float> tmp_res(maxt);
    for (size_t t = 0; t < maxt; t++) {
        size_t beg = t * n / maxt;
        size_t end = std::min(n, (t + 1) * n / maxt);
        tg1.run([&, t, beg, end] {
            float local_res = 0;
            for (size_t i = beg; i < end; i++) {
                local_res += std::sin(i);
            }
            tmp_res[t] = local_res;
        });
    }
    tg1.wait();
    for (size_t t = 0; t < maxt; t++) {
        tmp_res[t] += res;
        res = tmp_res[t];
    }
    tbb::task_group tg2;
    for (size_t t = 1; t < maxt; t++) {
        size_t beg = t * n / maxt - 1;
        size_t end = std::min(n, (t + 1) * n / maxt) - 1;
        tg2.run([&, t, beg, end] {
            float local_res = tmp_res[t];
            for (size_t i = beg; i < end; i++) {
                local_res += std::sin(i);
                a[i] = local_res;
            }
        });
    }
    tg2.wait();

    std::cout << a[n / 2] << std::endl;
    std::cout << res << std::endl;
    return 0;
}
