#include <cstdio>
using namespace std;

void modify(int *pa) {
    int *pb = pa + 1;  // 企图通过 a 的指针访问 b
    *pb = 3;           // 如果成功，那么 b 应该变成了 3
}

int func1() {
    int a = 1;
    int b = 2;
    modify(&a);
    return b;           // 时而 3，时而 2？
}

int func2() {
    struct {
        int a = 1;
        int b = 2;
    } s;
    modify(&s.a);
    return s.b;          // 3
}

int func3() {
    int a[2];
    a[0] = 1;
    a[1] = 2;
    modify(&a[0]);
    return a[1];         // 3
}

int main() {
    printf("%d\n", func3());
    return 0;
}
