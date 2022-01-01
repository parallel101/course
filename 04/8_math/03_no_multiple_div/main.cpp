void func(float *a, float b) {
    float inv_b = 1 / b;
    for (int i = 0; i < 1024; i++) {
        a[i] *= inv_b;
    }
}
