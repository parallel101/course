#include <iostream>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);
    float res = tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), (float)0,
    [&] (tbb::blocked_range<size_t> r, float local_res, auto is_final) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += std::sin(i);
            if (is_final) {
                a[i] = local_res;
            }
        }
        return local_res;
    }, [] (float x, float y) {
        return x + y;
    });

    std::cout << a[n / 2] << std::endl;
    std::cout << res << std::endl;
    return 0;
}
