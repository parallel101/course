#include <iostream>
#include <functional>

void call_twice(auto const &func) {
    std::cout << func(3.14f) << std::endl;
    std::cout << func(21) << std::endl;
}

int main() {
    auto twice = [] <class T> (T n) {
        return n * 2;
    };
    call_twice(twice);
    return 0;
}

/* 等价于：
auto twice(auto n) {
    return n * 2;
}
*/
