void func(int *a, int n) {
    n = n / 4 * 4;
    a = (int *)__builtin_assume_aligned(a, 16);
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
}
