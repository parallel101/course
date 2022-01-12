#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ticktock.h"

int main() {
    size_t n = 1<<24;
    std::vector<int> arr(n);
    std::generate(arr.begin(), arr.end(), std::rand);
    TICK(std_sort);
    std::sort(arr.begin(), arr.end(), std::less<int>{});
    TOCK(std_sort);
    return 0;
}
