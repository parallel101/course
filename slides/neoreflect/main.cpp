#include <string>
#include <iostream>
#include "reflect.hpp"

struct Z {
    std::string name;
    int age;
    int id;
};

int main() {
    Z z{"彭于斌", 23, 12345};
    std::cout << std::get<0>(reflect::reference(z)) << '\n'; // 第 0 个元素
    std::cout << std::get<1>(reflect::reference(z)) << '\n'; // 第 1 个元素
    std::cout << std::get<2>(reflect::reference(z)) << '\n'; // 第 2 个元素
}
