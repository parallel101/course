#include <iostream>
#include <vector>

void test_copy() {
    std::vector<int> v1(10);
    std::vector<int> v2(200);

    v1 = v2;                  // 拷贝赋值 O(n)

    std::cout << "after copy:" << std::endl;
    std::cout << "v1 length " << v1.size() << std::endl;  // 200
    std::cout << "v2 length " << v2.size() << std::endl;  // 200
}

void test_move() {
    std::vector<int> v1(10);
    std::vector<int> v2(200);

    v1 = std::move(v2);      // 移动赋值 O(1)

    std::cout << "after move:" << std::endl;
    std::cout << "v1 length " << v1.size() << std::endl;  // 200
    std::cout << "v2 length " << v2.size() << std::endl;  // 0
}

void test_swap() {
    std::vector<int> v1(10);
    std::vector<int> v2(200);

    std::swap(v1, v2);      // 交换两者 O(1)

    std::cout << "after swap:" << std::endl;
    std::cout << "v1 length " << v1.size() << std::endl;  // 200
    std::cout << "v2 length " << v2.size() << std::endl;  // 10
}

int main() {
    test_copy();
    test_move();
    test_swap();
    return 0;
}
