#include <cstdio>
#include <cstdint>

int main() {
    char a[4] = {1, 2, 3, 4};
    printf("0x%08x\n", *(int *)a);
    return 0;
}
