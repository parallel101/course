#include <iostream>
#include <vector>
#include <cmath>
#include "ticktock.h"

int main() {
    size_t n = 1<<24;

    std::vector<float> x(n);

    TICK(sin);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = std::sin(i);
    }
    TOCK(sin);

    TICK(sum);
    float ret = 0;
#pragma omp parallel for reduction(+:ret)
    for (int i = 0; i < n; i++) {
        ret += x[i];
    }
    TOCK(sum);

    std::cout << ret << std::endl;

    return 0;
}
