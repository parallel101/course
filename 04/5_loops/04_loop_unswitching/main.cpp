void func(float *__restrict a, float *__restrict b, bool is_mul) {
    for (int i = 0; i < 1024; i++) {
        if (is_mul) {
            a[i] = a[i] * b[i];
        } else {
            a[i] = a[i] + b[i];
        }
    }
}
