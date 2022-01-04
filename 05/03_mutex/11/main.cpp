#include <cstdio>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx;
    std::thread t1([&] {
        std::unique_lock grd(mtx);
        printf("t1 owns the lock\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    std::thread t2([&] {
        mtx.lock();
        std::unique_lock grd(mtx, std::adopt_lock);
        printf("t2 owns the lock\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    t1.join();
    t2.join();
    return 0;
}
