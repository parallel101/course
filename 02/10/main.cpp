#include <iostream>
#include <vector>

void test_copy() {
    std::cout << std::endl << "test copy:" << std::endl;

    std::vector<int> v1(10);
    std::vector<int> v2(200);

    std::cout << "before:" << std::endl;
    std::cout << "v1.size() is " << v1.size() << std::endl;
    std::cout << "v2.size() is " << v2.size() << std::endl;

    v2 = v1;

    std::cout << "after:" << std::endl;
    std::cout << "v1.size() is " << v1.size() << std::endl;
    std::cout << "v2.size() is " << v2.size() << std::endl;
}

void test_move() {
    std::cout << std::endl << "test move:" << std::endl;

    std::vector<int> v1(10);
    std::vector<int> v2(200);

    std::cout << "before:" << std::endl;
    std::cout << "v1.size() is " << v1.size() << std::endl;
    std::cout << "v2.size() is " << v2.size() << std::endl;

    v2 = std::move(v1);

    std::cout << "after:" << std::endl;
    std::cout << "v1.size() is " << v1.size() << std::endl;
    std::cout << "v2.size() is " << v2.size() << std::endl;
}

int main() {
    test_copy();
    test_move();
    return 0;
}
