#include <iostream>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <cmath>

int main() {
    size_t n = 1<<10;
    tbb::concurrent_vector<float> a(n);

    for (auto it = a.begin(); it != a.end(); ++it) {
        *it += 1.0f;
    }

    std::cout << a[1] << std::endl;

    return 0;
}
