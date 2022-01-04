#include <iostream>
#include <string>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx1;
    std::mutex mtx2;

    std::thread t1([&] {
        for (int i = 0; i < 1000; i++) {
            mtx1.lock();
            mtx1.unlock();
            mtx2.lock();
            mtx2.unlock();
        }
    });
    std::thread t2([&] {
        for (int i = 0; i < 1000; i++) {
            mtx2.lock();
            mtx2.unlock();
            mtx1.lock();
            mtx1.unlock();
        }
    });
    t1.join();
    t2.join();
    return 0;
}
