#include <iostream>
#include <tuple>

struct MyClass {
    int x;
    float y;
};

int main() {
    MyClass mc = {42, 3.14f};

    auto [x, y] = mc;

    std::cout << x << ", " << y << std::endl;
    return 0;
}
