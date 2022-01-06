#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

int main() {
    std::condition_variable cv;
    std::mutex mtx;

    std::thread t1([&] {
        std::unique_lock lck(mtx);
        cv.wait(lck);
        std::cout << "t1 is awake" << std::endl;
    });

    std::thread t2([&] {
        std::unique_lock lck(mtx);
        cv.wait(lck);
        std::cout << "t2 is awake" << std::endl;
    });

    std::thread t3([&] {
        std::unique_lock lck(mtx);
        cv.wait(lck);
        std::cout << "t3 is awake" << std::endl;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    std::cout << "notifying one" << std::endl;
    cv.notify_one();  // awakening t1 only

    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    std::cout << "notifying all" << std::endl;
    cv.notify_all();  // awakening t1 and t2

    t1.join();
    t2.join();
    t3.join();

    return 0;
}
