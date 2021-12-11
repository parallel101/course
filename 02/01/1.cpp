#include <vector>
#include <iostream>

int main() {
    std::vector<int> v;
    v.push_back(4);
    v.push_back(3);
    v.push_back(2);
    v.push_back(1);

    int sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i];
    }

    std::cout << sum << std::endl;
    return 0;
}
