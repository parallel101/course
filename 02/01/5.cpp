#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    std::vector<int> v = {4, 3, 2, 1};

    int sum = 0;
    std::for_each(v.begin(), v.end(), [&] (auto vi) {
        sum += vi;
    });

    std::cout << sum << std::endl;
    return 0;
}
