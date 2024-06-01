#include <cstdio>
#include <ranges>
using namespace std;

int main() {
    int a[2] = {1, 2};
    int *p = &a[0];
    printf("p     = %p\n", p);
    printf("p + 1 = %p\n", p + 1);
    printf("p + 2 = %p\n", p + 2);
    printf("%d\n", *p);
    printf("%d\n", *(p + 1));
    printf("%d\n", *(p + 2));

    return 0;
}
