#include <cstdio>
#include <cstdint>

int main() {
    int a[4] = {1, 2, 3, 4};
    int* p = &a[0];
    printf("之前: %p\n", p);
    p = p + 1;
    printf("之后: %p\n", p);
    printf("%d\n", *p);
    return 0;
}
