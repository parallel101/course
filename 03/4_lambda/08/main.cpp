#include <iostream>

template <class Func>
void call_twice(Func func) {
    std::cout << func(0) << std::endl;
    std::cout << func(1) << std::endl;
}

int main() {
    int fac = 2;
    int counter = 0;
    auto twice = [&] (int n) {
        counter++;
        return n * fac;
    };
    call_twice(twice);
    std::cout << "调用了 " << counter << " 次" << std::endl;
    return 0;
}
