#include <iostream>

template <class Func>
void call_twice(Func func) {
    std::cout << func(0) << std::endl;
    std::cout << func(1) << std::endl;
}

int main() {
    auto twice = [] (int n) -> int {
        return n * 2;
    };
    call_twice(twice);
    return 0;
}
