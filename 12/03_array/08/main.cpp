#include <cstdio>
#include <cstdint>

int main() {
    char a[4] = {1, 2, 3, 4};
    char* p = a;
    p = p + 2;
    printf("%d\n", *p);
    return 0;
}
