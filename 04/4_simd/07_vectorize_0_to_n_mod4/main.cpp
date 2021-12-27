void func(int *a, int n) {
    n = n / 4 * 4;
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
}
