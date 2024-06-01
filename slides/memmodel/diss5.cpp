#include <cstdio>
using namespace std;

[[gnu::noinline]] void modify(const int *p) {
    *(int *)p = 2;  // 企图去掉 const 来写入
}

[[gnu::noinline]] int test() {
    const int i = 1;
    modify(&i);
    return i;
}

int main() {
    printf("%d\n", test());
    return 0;
}
