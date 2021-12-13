#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight;

    Pig()
    {
        m_name = "佩奇";
        m_weight = 80;
    }
};

int main() {
    Pig pig;

    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;

    return 0;
}
