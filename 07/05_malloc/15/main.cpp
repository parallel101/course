#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ticktock.h"

float func(int n) {
    std::vector<float> tmp;
    for (int i = 0; i < n; i++) {
        tmp.push_back(i / 15 * 2.71828f);
    }
    std::reverse(tmp.begin(), tmp.end());
    float ret = tmp[32];
    return ret;
}

int main() {
    constexpr int n = 1<<25;

    TICK(first_call);
    std::cout << func(n) << std::endl;
    TOCK(first_call);

    TICK(second_call);
    std::cout << func(n - 1) << std::endl;
    TOCK(second_call);

    return 0;
}
