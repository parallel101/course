#include <cstdio>

void test(int &) {
    printf("int &\n");
}

void test(int const &) {
    printf("int const &\n");
}

void test(int &&) {
    printf("int &&\n");
}

int main() {
    int a = 0;
    int *p = &a;
    test(a);      // int &
    test(*p);     // int &
    test(p[a]);   // int &
    test(1);      // int &&
    test(a + 1);  // int &&
    test(*p + 1); // int &&

    const int b = 3;
    test(b);      // int const &
    test(b + 1);  // int &&
}
