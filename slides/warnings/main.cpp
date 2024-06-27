#include "func.h"
#include <array>
#include <cstdio>
#include <iostream>

struct Animal {
    virtual void speak() {
        printf("Animal 说话默认实现\n");
    }
};

struct Cat : Animal {
    void speak() override {
        printf("Cat 说：喵！\n");
    }
};

struct C {
    int x;
    int y;

    C() : x(2), y(x) {} // x y

    ~C() {
    } // ~y ~x
};

static int *func() {
    int *p = new int;
    return p;
}

int main() {
    C c1;
    C c2;
    std::cout << c1.y << '\n';

    int *p = func();
    std::cout << *p << '\n';
    delete p;
    std::array<int, 4> a{1, 2, 3, 4};
    int x = std::get<0>(a);
    printf("%d\n", x);
    Cat cat;
    cat.speak();
    func(BLUE);

    // size_t n;
    // std::cin >> n;
    // int s[n]; // Variable-length-array  MSVC 不支持
    // (void)s;

    for (unsigned int i = 1; i != 0; ++i)
        ; // 有符号整数溢出是 UB，无符号没问题

    return 0;
} // c2.~C(); c1.~C();
