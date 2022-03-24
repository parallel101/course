#include <cstdio>
#include <cstdint>

int main() {
    char a[4] = {1, 2, 3, 4};
    printf("%d\n", *(a + 2));
    return 0;
}
