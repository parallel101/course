#include <iostream>
#include <variant>

void print(std::variant<int, float> const &v) {
    std::visit([&] (auto const &t) {
        std::cout << t << std::endl;
    }, v);
}

auto add(std::variant<int, float> const &v1,
         std::variant<int, float> const &v2) {
    return std::visit([&] (auto const &t1, auto const &t2)
               -> std::variant<int, float> {
        return t1 + t2;
    }, v1, v2);
}

int main() {
    std::variant<int, float> v = 3;
    print(add(v, 3.14f));
    return 0;
}
