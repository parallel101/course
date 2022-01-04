#include <iostream>
#include <string>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx1;
    std::mutex mtx2;

    std::thread t1([&] {
        for (int i = 0; i < 1000; i++) {
            std::scoped_lock grd(mtx1, mtx2);
            // do something
        }
    });
    std::thread t2([&] {
        for (int i = 0; i < 1000; i++) {
            std::scoped_lock grd(mtx2, mtx1);
            // do something
        }
    });
    t1.join();
    t2.join();
    return 0;
}
