#include <iostream>
#include <tuple>

int main() {
    auto tup = std::tuple(3, 3.14f, 'h');

    auto [first, second, third] = tup;

    std::cout << first << std::endl;
    std::cout << second << std::endl;
    std::cout << third << std::endl;
    return 0;
}
