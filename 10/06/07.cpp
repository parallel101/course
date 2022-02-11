#include "bate.h"

#define N (1024*1024*256)

void test_float() {
    bate::timing("test_float");

    std::vector<float> arr(N);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        arr[i] = (i % 32) * 3.14f;
    }

    float ret = 0;
#pragma omp parallel for reduction(max:ret)
    for (int i = 0; i < N; i++) {
        float val = arr[i];
        ret = std::max(ret, val);
    }

    printf("%f\n", ret);

    bate::timing("test_float");
}

static int16_t ftoh(float f) {
    return *(int32_t *)&f >> 16;
}

static float htof(int16_t i) {
    int32_t l = i;
    l <<= 16;
    return *(float *)&l;
}

void test_bfloat16() {
    bate::timing("test_bfloat16");

    std::vector<int16_t> arr(N);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        arr[i] = ftoh((i % 32) * 3.14f);
    }

    float ret = 0;
#pragma omp parallel for reduction(max:ret)
    for (int i = 0; i < N; i++) {
        float val = htof(arr[i]);
        ret = std::max(ret, val);
    }

    printf("%f\n", ret);

    bate::timing("test_bfloat16");
}

int main() {
    test_float();
    test_bfloat16();
    return 0;
}
