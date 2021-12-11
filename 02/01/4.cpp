#include <vector>
#include <iostream>
#include <algorithm>

int sum = 0;

void func(int vi) {
    sum += vi;
}

int main() {
    std::vector<int> v = {4, 3, 2, 1};

    std::for_each(v.begin(), v.end(), func);

    std::cout << sum << std::endl;
    return 0;
}
