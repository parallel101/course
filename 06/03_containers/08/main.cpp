#include <iostream>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <cmath>

int main() {
    size_t n = 1<<10;
    tbb::concurrent_vector<float> a(n);

    tbb::parallel_for(tbb::blocked_range(a.begin(), a.end()),
    [&] (tbb::blocked_range<decltype(a.begin())> r) {
        for (auto it = r.begin(); it != r.end(); ++it) {
            *it += 1.0f;
        }
    });

    std::cout << a[1] << std::endl;

    return 0;
}
