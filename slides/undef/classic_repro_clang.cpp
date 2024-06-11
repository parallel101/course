#include <cstdio>

static void func() {
    printf("func called\n");
}

typedef void (*func_t)();

static func_t fp = nullptr;

extern void set_fp() {
    fp = func;
}

int main() {
    fp();
    return 0;
}
