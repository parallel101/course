#include <iostream>
#include <chrono>

void myfunc() {
    for (int i = 0; i < 100000; i++) {
        printf("Test!\n");
    }
}

void test_time(void func()) {
    auto t0 = std::chrono::steady_clock::now();
    func();
    auto t1 = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    std::cout << "Time elapsed: " << dt << "ms" << std::endl;
}

int main() {
    test_time(myfunc);
    return 0;
}
