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

static uint8_t ftoi(float f) {
    return unsigned(f * 2.f);
}

static float itof(uint8_t i) {
    return i * (1.f / 2.f);
}

void test_uint8() {
    bate::timing("test_uint8");

    std::vector<int8_t> arr(N);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        arr[i] = ftoi((i % 32) * 3.14f);
    }

    float ret = 0;
#pragma omp parallel for reduction(max:ret)
    for (int i = 0; i < N; i++) {
        float val = itof(arr[i]);
        ret = std::max(ret, val);
    }

    printf("%f\n", ret);

    bate::timing("test_uint8");
}

int main() {
    test_float();
    test_uint8();
    return 0;
}
