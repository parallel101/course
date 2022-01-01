void func(float *a, float b) {
    for (int i = 0; i < 1024; i++) {
        a[i] /= b;
    }
}
