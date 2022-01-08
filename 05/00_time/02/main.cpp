#include <iostream>
#include <thread>
#include <chrono>

int main() {
    auto t0 = std::chrono::steady_clock::now();
    for (volatile int i = 0; i < 10000000; i++);
    auto t1 = std::chrono::steady_clock::now();
    auto dt = t1 - t0;
    using double_ms = std::chrono::duration<double, std::milli>;
    double ms = std::chrono::duration_cast<double_ms>(dt).count();
    std::cout << "time elapsed: " << ms << " ms" << std::endl;
    return 0;
}
