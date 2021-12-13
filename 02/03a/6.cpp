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

void show(Pig pig) {
    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;
}

int main() {
    // show(80);    // 编译错误
    show(Pig(80));  // 编译通过

    return 0;
}
