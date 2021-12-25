#include <iostream>
#include <tuple>

int main() {
    auto tup = std::tuple<int, float, char>(3, 3.14f, 'h');

    int first = std::get<0>(tup);
    float second = std::get<1>(tup);
    char third = std::get<2>(tup);

    std::cout << first << std::endl;
    std::cout << second << std::endl;
    std::cout << third << std::endl;
    return 0;
}
