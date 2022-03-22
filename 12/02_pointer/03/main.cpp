#include <cstdio>
#include <cstdint>

int main() {
    int x = 1;
    int* p = &x;
    unsigned long address = (unsigned long)p;
    printf("%lu\n", address);
    return 0;
}
