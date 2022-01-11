#include <iostream>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<10;
    std::vector<float> a;
    std::vector<float *> pa(n);

    for (size_t i = 0; i < n; i++) {
        a.push_back(std::sin(i));
        pa[i] = &a.back();
    }

    for (size_t i = 0; i < n; i++) {
        std::cout << (&a[i] == pa[i]) << ' ';
    }
    std::cout << std::endl;

    return 0;
}
