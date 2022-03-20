#include <cmath>

void func(float *dst, float *src) {
    #pragma omp simd
    for (int i = 0; i < 4096; i++) {
        float r = src[i * 3];
        float g = src[i * 3 + 1];
        float b = src[i * 3 + 2];
        dst[i] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}
