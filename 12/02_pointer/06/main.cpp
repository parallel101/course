#include <cstdio>
#include <cstdint>

int func(int* psecond) {
    if (psecond != NULL)
        *psecond = 2;
    return 1;
}

int main() {
    int first = func(NULL);
    printf("first: %d\n", first);
    return 0;
}
