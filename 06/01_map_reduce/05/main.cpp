#include <iostream>
#include <tbb/parallel_for_each.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    tbb::parallel_for_each(a.begin(), a.end(), [&] (float &f) {
        f = 32.f;
    });

    return 0;
}
