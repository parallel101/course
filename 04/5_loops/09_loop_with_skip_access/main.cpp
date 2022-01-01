void func(float *a) {
    for (int i = 0; i < 1024; i++) {
        a[i * 2] += 1;
    }
}
