#include <cstdio>
#include <cstdint>

int main() {
    char str[] = {'H', 'e', 'l', 'l', 'o'};
    int len = sizeof(str);
    printf("%*s\n", len, str);
    return 0;
}
