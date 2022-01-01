void func(float *__restrict a, float *__restrict b, float dt) {
    for (int i = 0; i < 1024; i++) {
        a[i] = a[i] + b[i] * dt * dt;
    }
}
