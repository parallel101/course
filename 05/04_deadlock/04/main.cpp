#include <iostream>
#include <string>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx1;
    std::mutex mtx2;

    std::thread t1([&] {
        for (int i = 0; i < 1000; i++) {
            std::lock(mtx1, mtx2);
            mtx1.unlock();
            mtx2.unlock();
        }
    });
    std::thread t2([&] {
        for (int i = 0; i < 1000; i++) {
            std::lock(mtx2, mtx1);
            mtx2.unlock();
            mtx1.unlock();
        }
    });
    t1.join();
    t2.join();
    return 0;
}
