#include <cstdio>
#include <cstdint>
#include <cstring>

int main() {
    const char* str1 = "Hello, world!";
    const char* str2 = "Hello, world!";
    bool equal = str1 == str2;
    printf("%d\n", equal);
    return 0;
}
