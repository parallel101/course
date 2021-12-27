void func(float *a) {
#pragma GCC unroll 4
    for (int i = 0; i < 1024; i++) {
        a[i] = 1;
    }
}
