#include <iostream>

template <class Func>
void call_twice(Func const &func) {
    std::cout << func(0) << std::endl;
    std::cout << func(1) << std::endl;
    std::cout << "Func 大小: " << sizeof(Func) << std::endl;
}

auto make_twice(int fac) {
    return [=] (int n) {
        return n * fac;
    };
}

int main() {
    auto twice = make_twice(2);
    call_twice(twice);
    return 0;
}
