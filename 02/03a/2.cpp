#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight;

    Pig(std::string name, int weight) : m_name(name), m_weight(weight)
    {}
};

int main() {
    Pig pig("佩奇", 80);

    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;

    return 0;
}
