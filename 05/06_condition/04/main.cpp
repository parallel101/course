#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

int main() {
    std::condition_variable cv;
    std::mutex mtx;

    std::vector<int> foods;

    std::thread t1([&] {
        for (int i = 0; i < 2; i++) {
            std::unique_lock lck(mtx);
            cv.wait(lck, [&] {
                return foods.size() != 0;
            });
            auto food = foods.back();
            foods.pop_back();
            lck.unlock();

            std::cout << "t1 got food:" << food << std::endl;
        }
    });

    std::thread t2([&] {
        for (int i = 0; i < 2; i++) {
            std::unique_lock lck(mtx);
            cv.wait(lck, [&] {
                return foods.size() != 0;
            });
            auto food = foods.back();
            foods.pop_back();
            lck.unlock();

            std::cout << "t2 got food:" << food << std::endl;
        }
    });

    foods.push_back(42);
    foods.push_back(233);
    cv.notify_one();

    foods.push_back(666);
    foods.push_back(4399);
    cv.notify_all();

    t1.join();
    t2.join();

    return 0;
}
