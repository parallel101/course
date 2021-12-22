#include <iostream>
#include <vector>

template <class T>
T sum(std::vector<T> const &arr) {
    T res = 0;
    for (int i = 0; i < arr.size(); i++) {
        res += arr[i];
    }
    return res;
}

int main() {
    std::vector<int> a = {4, 3, 2, 1};
    std::cout << sum(a) << std::endl;
    std::vector<float> b = {3.14f, 2.718f};
    std::cout << sum(b) << std::endl;
}
