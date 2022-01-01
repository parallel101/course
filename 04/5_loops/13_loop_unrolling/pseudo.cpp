void func(float *a) {
    for (int i = 0; i < 1024; i += 4) {
        a[i + 0] = 1;
        a[i + 1] = 1;
        a[i + 2] = 1;
        a[i + 3] = 1;
    }
}
