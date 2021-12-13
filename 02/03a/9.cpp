#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight{0};
};

void show(Pig pig) {
    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;
}

int main() {
    Pig pig;

    show(pig);
    return 0;
}
