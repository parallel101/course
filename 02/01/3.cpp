#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {4, 3, 2, 1};

    int sum = 0;
    for (int vi: v) {
        sum += vi;
    }

    std::cout << sum << std::endl;
    return 0;
}
