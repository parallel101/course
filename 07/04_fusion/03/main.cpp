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
    for (size_t i = 0; i < n; i++) {
        a[i] = std::sin(i * 0.1f);
    }
    TICK(iter);
    for (int step = 0; step < steps; step++) {
#pragma omp parallel for
        for (size_t ibase = 16; ibase < n - 16; ibase += 32) {
            float ta[32 + 16 * 2], tb[32 + 16 * 2];
            for (intptr_t i = -16; i < 32 + 16; i++) {
                ta[16 + i] = a[ibase + i];
            }
            for (intptr_t s = 1; s < 16; s += 2) {
                for (intptr_t i = -16 + s; i < 32 + 16 - s; i++) {
                    tb[16 + i] = (ta[16 + i - 1] + ta[16 + i + 1]) * 0.5f;
                }
                for (intptr_t i = -16 + s + 1; i < 32 + 16 - s - 1; i++) {
                    ta[16 + i] = (tb[16 + i - 1] + tb[16 + i + 1]) * 0.5f;
                }
            }
            for (intptr_t i = 0; i < 32; i++) {
                b[ibase + i] = tb[16 + i];
            }
        }
        std::swap(a, b);
    }
    TOCK(iter);
    float loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (size_t i = 1; i < n - 1; i++) {
        loss += std::pow(a[i - 1] + a[i + 1] - a[i] * 2, 2);
    }
    loss = std::sqrt(loss);
    std::cout << "loss: " << loss << std::endl;
    return 0;
}
