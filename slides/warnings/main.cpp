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

static int test(int i) {
    // UB: 返回值不 void 的函数不写 return
    return i;
}

int main() {
    for (int i = 0; i < 13; ++i) {
        auto f = [] (int i) {
        };
        f(i + 1);
    }

    test(3);

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
    func(RED);

    // size_t n;
    // std::cin >> n;
    // int s[n]; // Variable-length-array  MSVC 不支持
    // (void)s;

    for (unsigned int i = 1; i != 0; ++i)
        ; // 有符号整数溢出是 UB，无符号没问题

    return 0;
} // c2.~C(); c1.~C();
