#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight;

    explicit Pig(int weight)
        : m_name("一只重达" + std::to_string(weight) + "kg的猪")
        , m_weight(weight)
    {}
};

int main() {
    // Pig pig = 80;  // 编译错误
    Pig pig(80);      // 编译通过

    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;

    return 0;
}
