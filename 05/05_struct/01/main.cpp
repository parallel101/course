#include <iostream>
#include <thread>
#include <mutex>

struct Counter {
    int m_count = 0;

    int get() const {
        return m_count;
    }

    void increase() {
        m_count++;
    }
};

Counter counter;

int main() {
    std::thread t1([&] () {
        for (int i = 0; i < 32; i++) {
            counter.increase();
        }
    });

    std::thread t2([&] () {
        for (int i = 0; i < 32; i++) {
            counter.increase();
        }
    });

    return 0;
}
