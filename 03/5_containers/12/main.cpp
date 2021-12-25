#include <iostream>
#include <variant>

void print(std::variant<int, float> const &v) {
    if (v.index() == 0) {
        std::cout << std::get<0>(v) << std::endl;
    } else if (v.index() == 1) {
        std::cout << std::get<1>(v) << std::endl;
    }
}

int main() {
    std::variant<int, float> v = 3;
    print(v);
    v = 3.14f;
    print(v);
    return 0;
}
