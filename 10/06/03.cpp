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

static int ftoi(float f) {
    return int(f * 100.f);
}

static float itof(int i) {
    return i * (1.f / 100.f);
}

void test_int32() {
    bate::timing("test_int32");

    std::vector<int32_t> arr(N);

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

    bate::timing("test_int32");
}

int main() {
    test_float();
    test_int32();
    return 0;
}
