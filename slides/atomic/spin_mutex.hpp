#pragma once

#include <atomic>

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
#if __cpp_lib_atomic_wait
        int retries = 1000;
        do {
            old = false;
            if (flag.compare_exchange_weak(old, true, std::memory_order_acquire, std::memory_order_relaxed))
                // load barrier
                return;
        } while (--retries);
#endif
        do {
#if __cpp_lib_atomic_wait
            flag.wait(true, std::memory_order_relaxed); // wait until flag != true
#endif
            old = false;
        } while (!flag.compare_exchange_weak(old, true, std::memory_order_acquire, std::memory_order_relaxed));
        // load barrier
    }

    void unlock() {
        // store barrier
        flag.store(false, std::memory_order_release);
#if __cpp_lib_atomic_wait
        flag.notify_one(); // flag changed, notify one of the waiters
#endif
    }

    std::atomic<bool> flag{false};
};
