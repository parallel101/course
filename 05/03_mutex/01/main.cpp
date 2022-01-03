#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main() {
    std::vector<int> arr;
    std::thread t1([&] {
        for (int i = 0; i < 1000; i++) {
            arr.push_back(1);
        }
    });
    std::thread t2([&] {
        for (int i = 0; i < 1000; i++) {
            arr.push_back(2);
        }
    });
    t1.join();
    t2.join();
    return 0;
}
