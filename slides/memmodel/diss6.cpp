#include <cstdio>
using namespace std;

[[gnu::noinline]] int func(int a, int b) {
    int *pa = &a;
    int *pb = pa - 1;
    return *pb;
}

int main() {
    printf("%d\n", func(1, 2));
    return 0;
}
