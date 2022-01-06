#include <iostream>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx;
    int counter = 0;

    std::thread t1([&] {
        for (int i = 0; i < 10000; i++) {
            mtx.lock();
            counter += 1;
            mtx.unlock();
        }
    });

    std::thread t2([&] {
        for (int i = 0; i < 10000; i++) {
            mtx.lock();
            counter += 1;
            mtx.unlock();
        }
    });

    t1.join();
    t2.join();

    std::cout << counter << std::endl;

    return 0;
}
