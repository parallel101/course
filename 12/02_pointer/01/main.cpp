#include <cstdio>
#include <cstdint>

int main() {
    int x = 1;
    int* p = &x;
    printf("x = %d\n", x);
    *p = 2;
    printf("x = %d\n", x);
    return 0;
}
