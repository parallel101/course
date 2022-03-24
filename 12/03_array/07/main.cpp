#include <cstdio>
#include <cstdint>

int main() {
    char a[4] = {1, 2, 3, 4};
    char* p = &a[0];
    p = p + 1;
    printf("%d\n", *p);
    return 0;
}
