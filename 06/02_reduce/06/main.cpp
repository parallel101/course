#include <iostream>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);
    float res = 0;

    for (size_t i = 0; i < n; i++) {
        res += std::sin(i);
        a[i] = res;
    }

    std::cout << a[n / 2] << std::endl;
    std::cout << res << std::endl;
    return 0;
}
