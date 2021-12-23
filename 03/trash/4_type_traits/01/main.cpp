#include <iostream>
#include <vector>

template <class T>
auto sum(T const &arr) {
    typename T::value_type res = 0;
    for (auto &&x: arr) {
        res += x;
    }
    return res;
}

int main() {
    std::vector<int> arr = {4, 3, 2, 1};
    auto res = sum(arr);
    std::cout << res << std::endl;
    return 0;
}
