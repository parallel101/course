#include <x86intrin.h>

float func(float *a) {
    __m128 ret = _mm_setzero_ps();
    for (int i = 0; i < 1024; i += 4) {
        __m128 a_i = _mm_loadu_ps(&a[i]);
        ret = _mm_add_ps(ret, a_i);
    }
    float r[4];
    _mm_storeu_ps(r, ret);
    return r[0] + r[1] + r[2] + r[3];
}
