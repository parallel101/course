int func(volatile int *a) {
    *a = 42;
    return *a;
}
