#include <iostream>
#include <tuple>

int main() {
    auto tup = std::tuple(3, 3.14f, 'h');

    auto &&[first, second, third] = tup;

    std::cout << std::get<0>(tup) << std::endl;
    first = 42;
    std::cout << std::get<0>(tup) << std::endl;

    return 0;
}
