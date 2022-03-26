#include <cstdio>
#include <cstdint>

int main() {
    unsigned char x = -128;
    short y = (short)x;
    printf("%d\n", y);
    return 0;
}
