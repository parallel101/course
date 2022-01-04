#include <cstdio>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx;
    std::thread t1([&] {
        std::unique_lock grd(mtx, std::defer_lock);
        std::lock_guard grd2(grd);
        printf("t1 owns the lock\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    std::thread t2([&] {
        std::unique_lock grd(mtx, std::defer_lock);
        std::lock_guard grd2(grd);
        printf("t2 owns the lock\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    t1.join();
    t2.join();
    return 0;
}
