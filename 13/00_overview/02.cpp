#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>

int main() {
    std::vector<int, std::allocator<int>> a = {1, 4, 2, 8, 5, 7};
    auto n = std::count_if(a.begin(), a.end(), std::bind2nd(std::less<int>(), 4));
    std::cout << n << std::endl;
    return 0;
}
