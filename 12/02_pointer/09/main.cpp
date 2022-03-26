#include <cstdio>
#include <cstdint>

int func(int* psecond) {
    if (psecond != nullptr)
        *psecond = 2;
    return 1;
}

int func(int second) {
    return 42;
}

int main() {
    int first = func(nullptr);
    printf("first: %d\n", first);
    return 0;
}
