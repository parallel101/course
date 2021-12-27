void func(int const *a, int const *b, int *__restrict c) {
    *c = *a;
    *c = *b;
}
