#include <atomic>
#include <iostream>
#include "parallel_pool.hpp"

/* bool compare_exchange_strong(int &flag, int &expected, int desired) { */
/*     if (flag == expected) { */
/*         flag = desired; */
/*         return true; */
/*     } else { */
/*         expected = flag; */
/*         return false; */
/*     } */
/* } */
/*  */
/* bool compare_exchange_weak(int &flag, int &expected, int desired) { */
/*     if (flag == expected && rand()) { */
/*         flag = desired; */
/*         return true; */
/*     } else { */
/*         expected = flag; */
/*         return false; */
/*     } */
/* } */

struct SpinMutex {
    void lock() {
        int expected;
        do
            expected = 0;
        while (!flag.compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed));
    }

    bool try_lock() {
        int expected = 0;
        if (!flag.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
            return false;
        } else {
            return true;
        }
    }

    void unlock() {
        flag.store(0, std::memory_order_release);
    }

    std::atomic<int> flag{0};
};

SpinMutex m;
int counter = 0;

void t1() {
    m.lock();
    counter++;
    m.unlock();
}

void t2() {
    m.lock();
    counter++;
    m.unlock();
}

int main() {
    ParallelPool pool{t1, t2};
    pool.join();
    std::cout << counter << '\n';
    return 0;
}
