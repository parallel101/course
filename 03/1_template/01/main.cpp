#include <iostream>

int twice(int t) {
    return t * 2;
}

float twice(float t) {
    return t * 2;
}

double twice(double t) {
    return t * 2;
}

int main() {
    std::cout << twice(21) << std::endl;
    std::cout << twice(3.14f) << std::endl;
    std::cout << twice(2.718) << std::endl;
}
