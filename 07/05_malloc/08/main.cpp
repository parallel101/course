#include <iostream>
#include <vector>
#include "ticktock.h"
#include "pod.h"

constexpr size_t n = 1ull<<32;  // 16GB

int main() {
    std::vector<pod<int>> arr(n);

    TICK(write_first);
    for (size_t i = 0; i < 1024; i++) {
        arr[i] = 1;
    }
    TOCK(write_first);
    return 0;
}
