#include <cstdio>

int main() {
    char c = 'a';
    printf("原字符：%c\n", c);
    c -= 'a';
    c += 'A';
    printf("转大写：%c\n", c);
    return 0;
}
