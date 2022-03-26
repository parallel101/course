#include <cstdio>
#include <cstdint>

int func(int* psecond) {
    if (psecond != 0)
        *psecond = 2;
    return 1;
}

int main() {
    int first = func(0);
    printf("first: %d\n", first);
    return 0;
}
