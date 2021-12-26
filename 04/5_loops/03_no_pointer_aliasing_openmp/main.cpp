void func(float *a, float *b) {
#pragma omp simd
    for (int i = 0; i < 1024; i++) {
        a[i] = b[i] + 1;
    }
}
