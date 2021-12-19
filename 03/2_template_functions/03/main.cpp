#include <iostream>

template <class T>
T twice(T t) {
    return t * 2;
}

int main() {
    std::cout << twice(21) << std::endl;
    std::cout << twice(3.14f) << std::endl;
    std::cout << twice(2.718) << std::endl;
}
