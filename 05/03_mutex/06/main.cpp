#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

int main() {
    std::vector<int> arr1;
    std::mutex mtx1;

    std::vector<int> arr2;
    std::mutex mtx2;

    std::thread t1([&] {
        for (int i = 0; i < 1000; i++) {
            {
                std::lock_guard grd(mtx1);
                arr1.push_back(1);
            }

            {
                std::lock_guard grd(mtx2);
                arr2.push_back(1);
            }
        }
    });
    std::thread t2([&] {
        for (int i = 0; i < 1000; i++) {
            {
                std::lock_guard grd(mtx1);
                arr1.push_back(2);
            }

            {
                std::lock_guard grd(mtx2);
                arr2.push_back(2);
            }
        }
    });
    t1.join();
    t2.join();
    return 0;
}
