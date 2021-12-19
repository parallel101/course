#include <cstdio>

int main() {
    int x = 233;
    int &ref = x;
    ref = 42;
    printf("%d\n", x);    // 42
    x = 1024;
    printf("%d\n", ref);  // 1024
}
