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

static int16_t ftoi(float f) {
    return int(f * 100.f);
}

static float itof(int16_t i) {
    return i * (1.f / 100.f);
}

void test_int16() {
    bate::timing("test_int16");

    std::vector<int16_t> arr(N);

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

    bate::timing("test_int16");
}

int main() {
    test_float();
    test_int16();
    return 0;
}
