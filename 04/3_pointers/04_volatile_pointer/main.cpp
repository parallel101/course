int func(int volatile *a) {
    *a = 42;
    return *a;
}
