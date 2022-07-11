#include <cstdio>

int main() {
    char s[] = "hello";
    printf("魔改前：%s\n", s);
    char *p = s + 3;
    printf("魔改后：%s\n", p);
}
