#include <iostream>

template <class T = int>
T two() {
    return 2;
}

int main() {
    std::cout << two<int>() << std::endl;
    std::cout << two<float>() << std::endl;
    std::cout << two<double>() << std::endl;
    std::cout << two() << std::endl;  // 等价于 two<int>()
}
