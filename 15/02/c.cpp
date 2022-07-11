#include <cstdio>

int main() {
    char s[] = "hello";
    printf("魔改前：%s\n", s);
    s[3] = 0;
    printf("魔改后：%s\n", s);
}
