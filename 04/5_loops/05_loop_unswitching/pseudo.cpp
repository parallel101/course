void func(float *__restrict a, float *__restrict b, bool is_mul) {
    if (is_mul) {
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] * b[i];
        }
    } else {
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] + b[i];
        }
    }
}
