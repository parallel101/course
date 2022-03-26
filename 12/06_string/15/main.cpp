#include <cstdio>
#include <cstdint>
#include <cstring>

int main() {
    char str1[] = "Hello, world!";
    char str2[] = "Hello, world!";
    bool equal = !strcmp(str1, str2);
    printf("%d\n", equal);
    return 0;
}
