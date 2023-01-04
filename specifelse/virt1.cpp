#include <cstdio>

struct A {
    virtual void hello() = 0;
    virtual void world() = 0;
};

struct B : A {
    virtual void hello() override { printf("B::hello\n"); }
    virtual void world() override { printf("B::world\n"); }
};

struct C : A {
    virtual void hello() override { printf("C::hello\n"); }
    virtual void world() override { printf("C::world\n"); }
};

int main() {
    A *p = new C;
    p->hello();
    p = new B;
    p->hello();
}
