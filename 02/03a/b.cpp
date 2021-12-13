#include <iostream>
#include <string>

struct Demo {
    explicit Demo(std::string a, std::string b) {
        std::cout << "Demo(" << a << ',' << b << ')' << std::endl;
    }
};

struct Pig {
    std::string m_name{"佩奇"};
    int m_weight = 80;
    Demo m_demo{"Hello", "world"};        // 编译通过
    // Demo m_demo = {"Hello", "world"};  // 编译出错
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
