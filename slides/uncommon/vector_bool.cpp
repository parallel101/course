#include <iostream>
#include <vector>

int main() {
    // vector<bool> 是特化过的，使用了位压缩，每个 bool 只占据 1 bit
    // 8 个 bool 才占用 1 个 char 的空间，从而节约了 8 倍存储空间
    std::vector<bool> v;
    v.push_back(true);
    v.push_back(false);
    v.push_back(true);
    v.push_back(false);

    for (size_t i = 0; i < v.size(); i++) {
        if (v[i]) {
            std::cout << "True" << '\n';
        } else {
            std::cout << "False" << '\n';
        }
    }

    // vector<bool> 无法实现合法的 data 函数
    // C++98: void data() {}
    // C++11: void data() = delete;
    // v.data();

    // [] 返回的是 std::_Bit_reference
    using T = decltype(v[0]);

    return 0;
}
