#include <cstdio>
#include <chrono>
#include <iostream>
#include <vector>

const size_t n = 64 * 1024 * 1024;

[[gnu::noinline]] void access(int *p) {
    for (size_t i = 0; i < n; ++i) {
        p[i] = i;
    }
}

int main() {
    std::vector<int> a(n + 1);
    for (size_t t = 0; t < 2; ++t) {
        for (size_t offset = 0; offset < 4; ++offset) {
            auto t0 = std::chrono::steady_clock::now();
            access((int *)((char *)a.data() + offset));
            access((int *)((char *)a.data() + offset));
            auto t1 = std::chrono::steady_clock::now();
            std::cout << offset << ": " << std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() << "s\n";
        }
    }
}
