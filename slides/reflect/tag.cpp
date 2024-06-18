#include <string>
#include <iostream>
#include "reflect.hpp"

struct X {
    std::string s;
};

struct Y {
    double a, b, c;
};

struct Z {
    int f1;
    long f2;
    size_t f3;
    std::string f4;
};

REFLECT_MEMBERS(Y, a, b, c);

int main() {
    std::cout << std::get<1>(reflect::name<Y>()).value << '\n';
    std::cout << reflect::count<X>() << '\n';
    std::cout << reflect::count<Y>() << '\n';
    std::cout << reflect::count<Z>() << '\n';
    std::cout << std::get<3>(reflect::reference(Z{5, 6, 7, "8h"})) << '\n';
}
