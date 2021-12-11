#include <vector>
#include <iostream>

int main() {
    std::vector<int> v(2);

    v[0] = 4;
    v[1] = 3;

    v.resize(4);

    v[2] = 2;
    v[3] = 1;

    int sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i];
    }

    std::cout << sum << std::endl;
    return 0;
}
