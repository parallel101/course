#include <cstdio>
#include <thread>
#include <mutex>

int main() {
    std::mutex mtx;
    std::thread t1([&] {
        std::unique_lock grd(mtx, std::try_to_lock);
        if (grd.owns_lock())
            printf("t1 success\n");
        else
            printf("t1 failed\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    std::thread t2([&] {
        std::unique_lock grd(mtx, std::try_to_lock);
        if (grd.owns_lock())
            printf("t2 success\n");
        else
            printf("t2 failed\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    });

    t1.join();
    t2.join();
    return 0;
}
