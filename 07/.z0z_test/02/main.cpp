#include <iostream>
#include <vector>
#include <cmath>
#include "ticktock.h"
#include <omp.h>

int main() {
    size_t n = 1<<29;

    std::vector<float> a(n);

    MTICK(scalar_fill, 10);
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
    MTOCK(scalar_fill);

    MTICK(vector_fill, 10);
#pragma omp simd
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
    MTOCK(vector_fill);

    return 0;
}
