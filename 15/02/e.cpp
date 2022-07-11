#include <cstdio>

int main() {
    char s[] = "hello\0world";
    printf("字符串：%s\n", s);
    char c = '\0';
    printf("字符：[%c]\n", c);
}
