#include <iostream>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <cmath>

int main() {
    size_t n = 1<<10;
    tbb::concurrent_vector<float> a;

    tbb::parallel_for((size_t)0, (size_t)n, [&] (size_t i) {
        auto it = a.grow_by(2);
        *it++ = std::cos(i);
        *it++ = std::sin(i);
    });

    std::cout << a.size() << std::endl;

    return 0;
}
