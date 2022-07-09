#include <cstdio>

struct Helper {
    Helper() {
        printf("before main\n");
    }

    ~Helper() {
        printf("after main\n");
    }
};

static Helper helper;

int main() {
    printf("inside main\n");
    return 0;
}
