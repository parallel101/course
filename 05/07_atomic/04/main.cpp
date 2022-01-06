#include <iostream>
#include <thread>
#include <atomic>

int main() {
    std::atomic<int> counter = 0;

    std::thread t1([&] {
        for (int i = 0; i < 10000; i++) {
            counter = counter + 1;
        }
    });

    std::thread t2([&] {
        for (int i = 0; i < 10000; i++) {
            counter = counter + 1;
        }
    });

    t1.join();
    t2.join();

    std::cout << counter << std::endl;

    return 0;
}
