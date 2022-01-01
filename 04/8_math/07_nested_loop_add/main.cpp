#include <cmath>

void func(float *a, float *b, float *c) {
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            c[i] += a[i] * b[j];
        }
    }
}
