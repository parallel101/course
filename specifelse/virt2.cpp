#include <cstdio>

struct A {
    void (*hello)(A *self) = nullptr;
    void (*world)(A *self) = nullptr;
};

void B_hello(A *self) { printf("B::hello\n"); }
void B_world(A *self) { printf("B::world\n"); }
struct B : A {
    B() {
        this->hello = B_hello;
        this->world = B_world;
    }
};

void C_hello(A *self) { printf("C::hello\n"); }
void C_world(A *self) { printf("C::world\n"); }
struct C : A {
    C() {
        this->hello = C_hello;
        this->world = C_world;
    }
};

int main() {
    A *p = new C;
    p->hello(p);
    p = new B;
    p->hello(p);
}
