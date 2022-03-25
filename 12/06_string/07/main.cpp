#include <cstdio>
#include <cstdint>
#include <cstring>

int main() {
    char str[6] = {'H', 'e', 'l', 'l', 'o', 0};
    printf("sizeof(str) = %ld\n", sizeof(str));
    printf("strlen(str) = %ld\n", strlen(str));
    return 0;
}
