#include <iostream>
#include <variant>

void print(std::variant<int, float> const &v) {
    if (std::holds_alternative<int>(v)) {
        std::cout << std::get<int>(v) << std::endl;
    } else if (std::holds_alternative<float>(v)) {
        std::cout << std::get<float>(v) << std::endl;
    }
}

int main() {
    std::variant<int, float> v = 3;
    print(v);
    v = 3.14f;
    print(v);
    return 0;
}
