#include "sumto.h"
#include <iostream>

int main() {
    constexpr bool debug = true;
    std::cout << sumto<debug>(4) << std::endl;
    return 0;
}
