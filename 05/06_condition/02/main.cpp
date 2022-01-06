#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

int main() {
    std::condition_variable cv;
    std::mutex mtx;
    bool ready = false;

    std::thread t1([&] {
        std::unique_lock lck(mtx);
        cv.wait(lck, [&] { return ready; });

        std::cout << "t1 is awake" << std::endl;
    });

    std::cout << "notifying not ready" << std::endl;
    cv.notify_one();  // useless now, since ready = false

    ready = true;
    std::cout << "notifying ready" << std::endl;
    cv.notify_one();  // awakening t1, since ready = true

    t1.join();

    return 0;
}
