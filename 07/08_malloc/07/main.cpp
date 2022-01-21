#include <iostream>
#include <vector>
#include "ticktock.h"

constexpr size_t n = 1ull<<32;  // 16GB

template <class T>
struct NoInit {
    T value;

    NoInit() { /* do nothing */ }
};

int main() {
    std::vector<NoInit<int>> arr(n);

    TICK(write_first);
    for (size_t i = 0; i < 1024; i++) {
        arr[i].value = 1;
    }
    TOCK(write_first);
    return 0;
}
