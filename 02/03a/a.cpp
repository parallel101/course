#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight{0};

    Pig(std::string name) : m_name(name)
    {}
};

void show(Pig pig) {
    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;
}

int main() {
    Pig pig("佩奇");

    show(pig);
    return 0;
}
