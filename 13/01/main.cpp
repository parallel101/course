#include "m128x3.h"

void func3(float *dst, float *src, int n) {
    n &= ~3;
    for (int i = 0; i < n; i += 4) {
        __m128 r = _mm_load_ps(src + i);
        __m128 g = _mm_load_ps(src + n + i);
        __m128 b = _mm_load_ps(src + n * 2 + i);
        __m128 tmp0 = r * 0.299f + g * 0.587f + b * 0.114f;
        _mm_store_ps(dst + i, tmp0);
    }
}

void func2(float *dst, float *src, int n) {
    n &= ~3;
    for (int i = 0; i < n; i += 4) {
        __m128x3 v = _mm128x3_transpose_ps(_mm128x3_load_ps(src + i * 3));
        __m128 tmp0 = v.val[0] * 0.299f + v.val[1] * 0.587f + v.val[2] * 0.114f;
        _mm_store_ps(dst + i, tmp0);
    }
}

void func1(float *dst, float *src, int n) {
    n &= ~3;
    for (int i = 0; i < n; i++) {
        float r = src[i * 3];
        float g = src[i * 3 + 1];
        float b = src[i * 3 + 2];
        dst[i] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}
