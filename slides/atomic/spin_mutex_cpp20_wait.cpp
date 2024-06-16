#include "mtpool.hpp"
using namespace std;

struct SpinMutex {
    bool try_lock() {
        bool old = false;
        if (flag.compare_exchange_strong(old, true, std::memory_order_acquire, std::memory_order_relaxed))
            // load barrier
            return true;
        return false;
    }

    void lock() {
        bool old;
        int retries = 1000;
        do {
            old = false;
            if (flag.compare_exchange_weak(old, true, std::memory_order_acquire, std::memory_order_relaxed))
                // load barrier
                return;
        } while (--retries);
        do {
            // fast-user-space mutex = futex (linux) SYS_futex
            flag.wait(true, std::memory_order_relaxed); // wait until not true
            old = false;
        } while (!flag.compare_exchange_weak(old, true, std::memory_order_acquire, std::memory_order_relaxed));
        // load barrier
    }

    void unlock() {
        // store barrier
        flag.store(false, std::memory_order_release);
        flag.notify_one();
    }

    std::atomic<bool> flag{false};
};

struct TestSpinMutex {
    SpinMutex mutex;

    void entry(MTIndex<0>) {
        mutex.lock();
        std::this_thread::sleep_for(1s);
        mutex.unlock();
    }

    void entry(MTIndex<1>) {
        mutex.lock();
        std::this_thread::sleep_for(1s);
        mutex.unlock();
    }
};

int main() {
    MTTest::runTest<TestSpinMutex>(1);
    return 0;
}
