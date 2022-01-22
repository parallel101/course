#include <iostream>
#include <cstdlib>
#include "ticktock.h"

constexpr size_t n = 1ull<<32;  // 16GB

int main() {
    int *arr = new int[n];

    TICK(write_first);
    for (size_t i = 0; i < 1024; i++) {
        arr[i] = 1;
    }
    TOCK(write_first);

    delete[] arr;
    return 0;
}
