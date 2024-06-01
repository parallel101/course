#include <cstddef>
#include <cstdio>
using namespace std;

struct A {
    int a;
};

struct B {
    int b;
};

struct C : A, B {
    int c;
};

int main() {
    C *c = new C;
    printf("C * = %p\n", c);
    printf("A * = %p\n", (A *)c);
    printf("B * = %p\n", (B *)c);
    printf("reinterpret B * = %p\n", reinterpret_cast<B *>(c));
    c->b = 4;
    printf("%d\n", reinterpret_cast<B *>(c)->b);
    printf("%d\n", static_cast<B *>(c)->b);
    return 0;
}
