#include <iostream>
#include <chrono>

using namespace std::literals;

int main() {
    std::cout << (1s + 100ms).count() << std::endl;
}
