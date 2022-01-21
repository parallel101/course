#include <iostream>
#include <vector>
#include "ticktock.h"

constexpr size_t n = 1<<29;

int main() {
    std::vector<int> arr(n);

    TICK(write_0);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 0;
    }
    TOCK(write_0);

    TICK(write_1);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1;
    }
    TOCK(write_1);

    return 0;
}
