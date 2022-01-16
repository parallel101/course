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
            for (intptr_t s = 2; s <= HS; s += 4) {
#pragma omp simd
                for (intptr_t i = -HS + 2; i < BS + HS - 2; i++) {
                    tb[HS + i] = (ta[HS + i - 2] + ta[HS + i] + ta[HS + i] + ta[HS + i + 2]) * 0.25f;
                }
#pragma omp simd
                for (intptr_t i = -HS + 4; i < BS + HS - 4; i++) {
                    ta[HS + i] = (tb[HS + i - 2] + tb[HS + i] + tb[HS + i] + tb[HS + i + 2]) * 0.25f;
                }
            }
            for (intptr_t i = 0; i < BS; i += 4) {
                _mm_stream_ps(&b[ibase + i], _mm_loadu_ps(&tb[HS + i]));
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
