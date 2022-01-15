#include <iostream>
#include <variant>
#include <vector>
#include <memory>

struct A {
    void func() {
        printf("A::func\n");
    }
};

struct B {
    void func() {
        printf("B::func\n");
    }
};

struct AnyBase {
    virtual void func() = 0;
};

template <class T>
struct AnyImpl : AnyBase {
    T t;

    AnyImpl(T t) : t(std::move(t)) {}

    void func() override {
        t.func();
    }
};

struct Any {
    std::unique_ptr<AnyBase> ptr;

    template <class T>
    Any(T t) : ptr(std::make_unique<AnyImpl<T>>(std::move(t))) {}

    template <class T>
    T *cast() {
        auto p = dynamic_cast<AnyImpl<T> *>(ptr.get());
        return p ? &p->t : nullptr;
    }

    void func() {
        ptr->func();
    }
};

int main() {
    Any c = A{};
    c.func();
    c = B{};
    c.func();
    A *ap = c.cast<A>();
    return 0;
}
