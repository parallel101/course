#include <iostream>
#include <string>

struct Pig {
    std::string m_name;
    int m_weight;

    explicit Pig(std::string name, int weight)
        : m_name(name)
        , m_weight(weight)
    {}
};

void show(Pig pig) {
    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;
}

Pig func1() {
    return {"佩奇", 80};        // 编译错误
}

Pig func2() {
    return Pig{"佩奇", 80};     // 编译通过
}

Pig func3() {
    return Pig("佩奇", 80);     // 编译通过
}

int main() {
    Pig pig1 = {"佩奇", 80};    // 编译错误
    Pig pig2{"佩奇", 80};       // 编译通过
    Pig pig3("佩奇", 80);       // 编译通过

    show({"佩奇", 80});         // 编译错误
    show(Pig("佩奇", 80));      // 编译通过

    return 0;
}
