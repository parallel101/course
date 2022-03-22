#include <cstdio>
#include <cstdint>

int main() {
    int x = 0x12345678;
    int* p = &x;
    char* pc = (char*)p;
    printf("%x\n", *pc);
    return 0;
}
