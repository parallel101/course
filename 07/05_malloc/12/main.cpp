#include <iostream>
#include <vector>
#include "ticktock.h"
#include <cstdlib>

constexpr size_t n = 1<<20;

int main() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; i++) {
        std::vector<int> arr(n);
        bool is_aligned = (uintptr_t)arr.data() % 16 == 0;
        std::cout << "std: " << is_aligned << std::endl;
    }
    for (int i = 0; i < 5; i++) {
        auto arr = (int *)malloc(n * sizeof(int));
        bool is_aligned = (uintptr_t)arr % 16 == 0;
        std::cout << "malloc: " << is_aligned << std::endl;
        free(arr);
    }
    return 0;
}
