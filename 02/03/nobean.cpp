#include <fstream>
#include <iostream>
#include <stdexcept>

void test() {
    std::ofstream fout("a.txt");
    fout << "有一种病\n";
    throw std::runtime_error("中道崩殂");
    fout << "叫 JavaBean\n";
}

int main() {
    try {
        test();
    } catch (std::exception const &e) {
        std::cout << "捕获异常：" << e.what() << std::endl;
    }
    return 0;
}
