#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight;
};

void show(Pig pig) {
    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;
}

int main() {
    Pig pig1 = {"佩奇", 80};   // 编译通过
    Pig pig2{"佩奇", 80};      // 编译通过
    // Pig pig3("佩奇", 80);   // 编译错误！

    show(pig1);
    return 0;
}
