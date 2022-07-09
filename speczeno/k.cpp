#include <cstdio>

struct Helper {
    Helper() {
        printf("helper initialized\n");
    }

    ~Helper() {
        printf("helper destroyed\n");
    }
};

static void before_main() {
    static Helper helper;
    printf("before main\n");
}

static int globalHelper = (before_main(), 0);

int main() {
    printf("inside main");
    return 0;
}

