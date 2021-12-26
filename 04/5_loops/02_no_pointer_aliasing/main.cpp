void func(float *__restrict a, float *__restrict b) {
    for (int i = 0; i < 1024; i++) {
        a[i] = b[i] + 1;
    }
}
