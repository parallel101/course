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

void test_double() {
    bate::timing("test_double");

    std::vector<double> arr(N);

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        arr[i] = (i % 32) * 3.14;
    }

    double ret = 0;
#pragma omp parallel for reduction(max:ret)
    for (int i = 0; i < N; i++) {
        double val = arr[i];
        ret = std::max(ret, val);
    }

    printf("%lf\n", ret);

    bate::timing("test_double");
}

int main() {
    test_double();
    test_float();
    return 0;
}
