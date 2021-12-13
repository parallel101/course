#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight;

    Pig(int weight)
        : m_name("一只重达" + std::to_string(weight) + "kg的猪")
        , m_weight(weight)
    {}
};

int main() {
    Pig pig = 80;

    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;

    return 0;
}
