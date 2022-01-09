#include <iostream>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    for (size_t i = 0; i < n; i++) {
        a[i] = std::sin(i);
    }

    return 0;
}
