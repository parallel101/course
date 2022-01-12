#include <iostream>
#include <vector>
#include <cmath>
#include "ticktock.h"
#include <omp.h>

int main() {
    size_t n = 1<<29;

    std::vector<float> a(n);

    TICK(serial_add);
    for (int i = 0; i < n; i++) {
        a[i] = a[i] + 1;
    }
    TOCK(serial_add);

    TICK(parallel_add);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = a[i] + 1;
    }
    TOCK(parallel_add);

    TICK(serial_sqrmuladd);
    for (int i = 0; i < n; i++) {
        a[i] = a[i] * a[i] * 3.14f + 1;
    }
    TOCK(serial_sqrmuladd);

    TICK(parallel_sqrmuladd);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = a[i] * a[i] * 3.14f + 1;
    }
    TOCK(parallel_sqrmuladd);

    TICK(serial_div);
    for (int i = 0; i < n; i++) {
        a[i] = 3.14f / a[i];
    }
    TOCK(serial_div);

    TICK(parallel_div);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = 3.14f / a[i];
    }
    TOCK(parallel_div);

    return 0;

    TICK(serial_sqrt);
    for (int i = 0; i < n; i++) {
        a[i] = std::sqrt(a[i]);
    }
    TOCK(serial_sqrt);

    TICK(parallel2_sqrt);
    omp_set_num_threads(2);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = std::sqrt(a[i]);
    }
    TOCK(parallel2_sqrt);

    TICK(parallel4_sqrt);
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = std::sqrt(a[i]);
    }
    TOCK(parallel4_sqrt);

    TICK(parallel8_sqrt);
    omp_set_num_threads(8);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = std::sqrt(a[i]);
    }
    TOCK(parallel8_sqrt);

    TICK(serial_sin);
    for (int i = 0; i < n; i++) {
        a[i] = std::sin(a[i]);
    }
    TOCK(serial_sin);

    TICK(parallel2_sin);
    omp_set_num_threads(2);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = std::sin(a[i]);
    }
    TOCK(parallel2_sin);

    TICK(parallel4_sin);
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = std::sin(a[i]);
    }
    TOCK(parallel4_sin);

    TICK(parallel8_sin);
    omp_set_num_threads(8);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = std::sin(a[i]);
    }
    TOCK(parallel8_sin);

    return 0;
}
