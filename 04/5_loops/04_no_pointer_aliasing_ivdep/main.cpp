void func(float *a, float *b) {
#pragma GCC ivdep
    for (int i = 0; i < 1024; i++) {
        a[i] = b[i] + 1;
    }
}
