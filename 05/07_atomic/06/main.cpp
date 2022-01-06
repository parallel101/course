#include <iostream>
#include <thread>
#include <atomic>

int main() {
    std::atomic<int> counter;
    counter.store(0);

    std::thread t1([&] {
        for (int i = 0; i < 10000; i++) {
            int old = counter.exchange(1);
            if (old != 1)
                std::cout << "old=" << old << std::endl;
        }
    });

    std::thread t2([&] {
        for (int i = 0; i < 10000; i++) {
            int old = counter.exchange(2);
            if (old != 2)
                std::cout << "old=" << old << std::endl;
        }
    });

    t1.join();
    t2.join();

    std::cout << "counter=" << counter.load() << std::endl;

    return 0;
}
