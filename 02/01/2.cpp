#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {4, 3, 2, 1};

    int sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i];
    }

    std::cout << sum << std::endl;
    return 0;
}
