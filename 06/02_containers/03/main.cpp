#include <iostream>
#include <vector>
#include <tbb/concurrent_vector.h>
#include <cmath>

int main() {
    size_t n = 1<<10;
    tbb::concurrent_vector<float> a;
    std::vector<float *> pa(n);

    for (size_t i = 0; i < n; i++) {
        auto it = a.push_back(std::sin(i));
        pa[i] = &*it;
    }

    for (size_t i = 0; i < n; i++) {
        std::cout << (&a[i] == pa[i]) << ' ';
    }
    std::cout << std::endl;

    return 0;
}
