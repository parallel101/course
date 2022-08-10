#include <iostream>
#include <thread>
#include <vector>

int main() {
    std::vector<int> arr;

    std::thread t1([&] () {
        for (int i = 0; i < 1000; i++) {
            arr.push_back(i);
        }
    });

    std::thread t2([&] () {
        for (int i = 0; i < 1000; i++) {
            arr.push_back(1000 + i);
        }
    });

    t1.join();
    t2.join();

    std::cout << arr.size() << std::endl;

    return 0;
}
