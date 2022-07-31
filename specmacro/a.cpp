#include <iostream>
#include <string>
#include "scienum.h"

enum Color {
    RED = 1, GREEN = 2, BLUE = 3, YELLOW = 4,
};

int main() {
    std::cout << scienum::get_enum_name(YELLOW) << std::endl;
    std::cout << scienum::enum_from_name<Color>("YELLOW") << std::endl;
}
