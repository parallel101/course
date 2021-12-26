void func(int *a) {
    __m128i curr = {0, 1, 2, 3};
    __m128i delta = {4, 4, 4, 4};
    for (int i = 0; i < 1024; i += 4) {
        a[i : i + 4] = curr;
        curr += delta;
    }
}
