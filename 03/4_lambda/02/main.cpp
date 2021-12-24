#include <iostream>
#include <chrono>

void myfunc() {
    for (int i = 0; i < 1000000; i++) {
        printf("Test!\n");
    }
}

int main() {
    auto t0 = std::chrono::steady_clock::now();
    myfunc();
    auto t1 = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    std::cout << "Time elapsed: " << dt << "ms" << std::endl;
    return 0;
}
