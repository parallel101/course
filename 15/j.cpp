#include <cstdio>

static void func() {
    static int helper = printf("helper initialized\n");
    printf("inside func\n");
}

int main() {
    printf("first time call func\n"); func();
    printf("second time call func\n"); func();
    printf("third time call func\n"); func();
    return 0;
}
