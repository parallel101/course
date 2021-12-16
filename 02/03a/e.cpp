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
    Pig pig{"佩奇", 80};

    show(pig);

    Pig pig2 = pig;    // 调用 Pig(Pig const &)
    // Pig pig2(pig);  // 与上一种方式等价

    show(pig);

    return 0;
}
