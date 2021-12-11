#include <fmt/core.h>
#include <iostream>

int main() {
    std::string msg = fmt::format("The answer is {}.\n", 42);
    std::cout << msg << std::endl;
    return 0;
}
