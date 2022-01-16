#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include "ticktock.h"
#include "mtprint.h"
#include <x86intrin.h>
#include <omp.h>

// L1: 32KB
// L2: 256KB
// L3: 12MB

constexpr size_t n = 1<<26;
constexpr size_t steps = 32/16;

std::vector<float> a(n);  // 256MB
std::vector<float> b(n);

int main() {
#pragma omp parallel for
    for (intptr_t i = 0; i < n; i++) {
        a[i] = std::sin(i * 0.1f);
    }
    TICK(iter);
    for (int step = 0; step < steps; step++) {
        constexpr intptr_t BS = 128;
        constexpr intptr_t HS = 16;
#pragma omp parallel for
        for (intptr_t ibase = HS; ibase < n - HS; ibase += BS) {
            float ta[BS + HS * 2], tb[BS + HS * 2];
            for (intptr_t i = -HS; i < BS + HS; i++) {
                ta[HS + i] = a[ibase + i];
            }
#pragma GCC unroll 8
            for (intptr_t s = 1; s < HS; s += 2) {
#pragma omp simd
                for (intptr_t i = -HS + s; i < BS + HS - s; i++) {
                    tb[HS + i] = (ta[HS + i - 1] + ta[HS + i + 1]) * 0.5f;
                }
#pragma omp simd
                for (intptr_t i = -HS + s + 1; i < BS + HS - s - 1; i++) {
                    ta[HS + i] = (tb[HS + i - 1] + tb[HS + i + 1]) * 0.5f;
                }
            }
            for (intptr_t i = 0; i < BS; i++) {
                b[ibase + i] = tb[HS + i];
            }
        }
        std::swap(a, b);
    }
    TOCK(iter);
    float loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (intptr_t i = 1; i < n - 1; i++) {
        loss += std::pow(a[i - 1] + a[i + 1] - a[i] * 2, 2);
    }
    loss = std::sqrt(loss);
    std::cout << "loss: " << loss << std::endl;
    return 0;
}
