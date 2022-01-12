#include <iostream>
#include <vector>
#include <cmath>
#include "ticktock.h"

int main() {
    size_t n = 1<<26;

    std::vector<float> a(n);

    TICK(step1);
    for (int i = 0; i < n; i++) {
        a[i] = std::sin(i);
    }
    TOCK(step1);

    TICK(step2);
    float ret = 0;
    for (int i = 0; i < n; i++) {
        ret += a[i];
    }
    TOCK(step2);

    std::cout << ret << std::endl;

    return 0;
}
