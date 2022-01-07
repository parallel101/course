#include <iostream>
#include <thread>
#include <chrono>

int main() {
    auto t0 = std::chrono::steady_clock::now();
    for (volatile int i = 0; i < 100000; i++);
    auto t1 = std::chrono::steady_clock::now();
    auto dt = t1 - t0;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
    std::cout << "time elapsed: " << ms << " ms" << std::endl;
    return 0;
}
