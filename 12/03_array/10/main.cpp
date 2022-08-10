#include <cstdio>
#include <cstdint>

int main() {
    int a[4] = {1, 2, 3, 4};
    int* p = &a[0];
    p = p + 1;
    printf("%d\n", *p);
    return 0;
}
