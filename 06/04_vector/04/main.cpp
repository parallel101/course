#include <iostream>
#include <tbb/concurrent_vector.h>
#include <cmath>

int main() {
    size_t n = 1<<10;
    tbb::concurrent_vector<float> a;

    for (size_t i = 0; i < n; i++) {
        auto it = a.grow_by(2);
        *it++ = std::cos(i);
        *it++ = std::sin(i);
    }

    std::cout << a.size() << std::endl;

    return 0;
}
