#include <iostream>
#include <vector>
#include "ticktock.h"
#include "alignalloc.h"

constexpr size_t n = 1<<20;

int main() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; i++) {
        std::vector<int, AlignedAllocator<int>> arr(n);
        bool is_aligned = (uintptr_t)arr.data() % 64 == 0;
        std::cout << "64: " << is_aligned << std::endl;
    }
    for (int i = 0; i < 5; i++) {
        std::vector<int, AlignedAllocator<int, 4096>> arr(n);
        bool is_aligned = (uintptr_t)arr.data() % 4096 == 0;
        std::cout << "4096: " << is_aligned << std::endl;
    }
    return 0;
}
