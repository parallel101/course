#include <iostream>
#include <functional>

void call_twice(int func(int)) {
    std::cout << func(0) << std::endl;
    std::cout << func(1) << std::endl;
    std::cout << "Func 大小: " << sizeof(func) << std::endl;
}

int main() {
    call_twice([] (auto n) {
        return n * 2;
    });
    return 0;
}
