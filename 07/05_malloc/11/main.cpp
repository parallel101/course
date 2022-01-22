#include <iostream>
#include <vector>
#include "ticktock.h"
#include <tbb/cache_aligned_allocator.h>

constexpr size_t n = 1<<20;

int main() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; i++) {
        std::vector<int> arr(n);
        bool is_aligned = (uintptr_t)arr.data() % 64 == 0;
        std::cout << "std: " << is_aligned << std::endl;
    }
    for (int i = 0; i < 5; i++) {
        std::vector<int, tbb::cache_aligned_allocator<int>> arr(n);
        bool is_aligned = (uintptr_t)arr.data() % 64 == 0;
        std::cout << "tbb: " << is_aligned << std::endl;
    }
    return 0;
}
