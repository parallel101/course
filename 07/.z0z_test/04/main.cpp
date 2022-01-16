#include <iostream>
#include <vector>
#include <cmath>
#include "ticktock.h"

int main() {
    size_t n = 1<<29;

    std::vector<float> a(n);

    TICK(serial);
    for (int i = 0; i < n; i += 1) {
        a[i] = 1;
    }
    TOCK(serial);

    TICK(step1);
#pragma omp parallel for
    for (int i = 0; i < n; i += 1) {
        a[i] = 1;
    }
    TOCK(step1);

    TICK(step2);
#pragma omp parallel for
    for (int i = 0; i < n; i += 2) {
        a[i] = 1;
    }
    TOCK(step2);

    TICK(step4);
#pragma omp parallel for
    for (int i = 0; i < n; i += 4) {
        a[i] = 1;
    }
    TOCK(step4);

    TICK(step8);
#pragma omp parallel for
    for (int i = 0; i < n; i += 8) {
        a[i] = 1;
    }
    TOCK(step8);

    TICK(step16);
#pragma omp parallel for
    for (int i = 0; i < n; i += 16) {
        a[i] = 1;
    }
    TOCK(step16);

    TICK(step32);
#pragma omp parallel for
    for (int i = 0; i < n; i += 32) {
        a[i] = 1;
    }
    TOCK(step32);

    TICK(step64);
#pragma omp parallel for
    for (int i = 0; i < n; i += 64) {
        a[i] = 1;
    }
    TOCK(step64);

    TICK(step128);
#pragma omp parallel for
    for (int i = 0; i < n; i += 128) {
        a[i] = 1;
    }
    TOCK(step128);

    return 0;
}
