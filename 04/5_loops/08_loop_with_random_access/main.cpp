void func(float *a, int *b) {
    for (int i = 0; i < 1024; i++) {
        a[b[i]] += 1;
    }
}
