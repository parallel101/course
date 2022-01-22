#include <iostream>
#include <vector>
#include "ticktock.h"
#include <cstdlib>
#include <x86intrin.h>

constexpr size_t n = 1<<20;

int main() {
    std::cout << std::boolalpha;
    for (int i = 0; i < 5; i++) {
        auto arr = (int *)_mm_malloc(n * sizeof(int), 4096);
        bool is_aligned = (uintptr_t)arr % 4096 == 0;
        std::cout << "_mm_malloc: " << is_aligned << std::endl;
        _mm_free(arr);
    }
    for (int i = 0; i < 5; i++) {
        auto arr = (int *)aligned_alloc(4096, n * sizeof(int));
        bool is_aligned = (uintptr_t)arr % 4096 == 0;
        std::cout << "aligned_alloc: " << is_aligned << std::endl;
        free(arr);
    }
    return 0;
}
