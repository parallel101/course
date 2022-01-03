#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

int main() {
    std::vector<int> arr;
    std::mutex mtx;
    std::thread t1([&] {
        for (int i = 0; i < 1000; i++) {
            std::unique_lock grd(mtx);
            arr.push_back(1);
        }
    });
    std::thread t2([&] {
        for (int i = 0; i < 1000; i++) {
            std::unique_lock grd(mtx);
            arr.push_back(2);
            grd.unlock();
            printf("outside of lock\n");
            // grd.lock();  // 如果需要，还可以重新上锁
        }
    });
    t1.join();
    t2.join();
    return 0;
}
