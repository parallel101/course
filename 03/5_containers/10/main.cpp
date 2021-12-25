#include <iostream>
#include <variant>

int main() {
    std::variant<int, float> v = 3;

    std::cout << std::get<int>(v) << std::endl;   // 3
    std::cout << std::get<0>(v) << std::endl;     // 3

    v = 3.14f;

    std::cout << std::get<float>(v) << std::endl; // 3.14f
    std::cout << std::get<int>(v) << std::endl;   // 运行时错误

    return 0;
}
