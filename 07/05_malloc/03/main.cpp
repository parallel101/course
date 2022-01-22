#include <iostream>
#include <cstdlib>
#include "ticktock.h"

constexpr size_t n = 1<<29;

int main() {
    int *arr = (int *)malloc(n * sizeof(int));

    TICK(write_first);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1;
    }
    TOCK(write_first);

    TICK(write_second);
    for (size_t i = 0; i < n; i++) {
        arr[i] = 1;
    }
    TOCK(write_second);

    free(arr);
    return 0;
}
