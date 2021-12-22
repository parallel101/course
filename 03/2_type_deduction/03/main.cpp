#include <cstdio>

int main() {
    int x = 233;
    int const &ref = x;
    // ref = 42;  // 会出错！
    printf("%d\n", x);    // 233
    x = 1024;
    printf("%d\n", ref);  // 1024
}
