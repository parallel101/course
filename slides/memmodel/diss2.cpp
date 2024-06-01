#include <cstdio>
using namespace std;

[[gnu::noinline]] void modify(void *p) {
    *(short *)p = 3;               // 企图修改低 16 位
}

[[gnu::noinline]] int func1() {
    int i = 0x10002;
    modify((void *)&i);
    return i;
}

int main() {
    printf("%#x", func1());
    return 0;
}
